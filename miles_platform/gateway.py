"""Authenticated inference with transactional rate limits and observed usage."""

import asyncio
import json
from decimal import Decimal

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from miles_platform.store import now, uid
from miles_platform import billing


class Chat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "miles-default"
    messages: list[dict] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    stream: bool = False
    stream_options: dict | None = None

    @model_validator(mode="after")
    def validate_messages(self):
        size = 0
        for message in self.messages:
            if set(message) - {"role", "content"} or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Only text system, user and assistant messages are supported")
            if not isinstance(message.get("content"), str):
                raise ValueError("Message content must be text")
            size += len(message["content"])
        if size > 16000:
            raise ValueError("Messages exceed the 16000-character request limit")
        return self


def reserve(store, key, deployment):
    request_id = uid("req")
    billing.prepare(store)
    rates = billing.pricing(store)
    with store.connect(write=True) as db:
        if key["id"] != "console":
            current = db.execute("SELECT revoked FROM api_keys WHERE id=?", (key["id"],)).fetchone()
            if not current or current["revoked"]:
                raise HTTPException(401, "API key was revoked")
        count = db.execute(
            "SELECT COUNT(*) FROM usage WHERE key_id=? AND created>?", (key["id"], now() - 60)
        ).fetchone()[0]
        if count >= key["rpm"]:
            raise HTTPException(429, "Request limit reached; retry in one minute", headers={"Retry-After": "60"})
        concurrent = db.execute("SELECT COUNT(*) FROM usage WHERE ended IS NULL").fetchone()[0]
        if concurrent >= 2:
            raise HTTPException(429, "Workspace concurrency limit reached (2)")
        billing.reserve(db, store, request_id, "inference", billing.inference_quote(store, rates))
        db.execute(
            """INSERT INTO usage(id,key_id,deployment_id,created,status,input_rate,output_rate)
                    VALUES (?,?,?,?,102,?,?)""",
            (request_id, key["id"], deployment["id"], now(), rates.get("input_million"), rates.get("output_million")),
        )
    return request_id


def record(store, request_id, status, usage):
    measured = usage is not None and "prompt_tokens" in usage and "completion_tokens" in usage
    prompt = max(0, int((usage or {}).get("prompt_tokens", 0)))
    completion = max(0, int((usage or {}).get("completion_tokens", 0)))
    row = store.one("SELECT input_rate,output_rate FROM usage WHERE id=?", (request_id,))
    cost = None
    if measured and row["input_rate"] is not None and row["output_rate"] is not None:
        cost = float(
            (prompt * Decimal(str(row["input_rate"])) + completion * Decimal(str(row["output_rate"]))) / 1_000_000
        )
    with store.connect(write=True) as db:
        db.execute(
            "UPDATE usage SET ended=?,status=?,prompt_tokens=?,completion_tokens=?,complete=?,cost=? WHERE id=?",
            (now(), status, prompt, completion, int(measured), cost, request_id),
        )
        billing.settle(db, request_id, billing.micros(cost) if cost is not None else 0)


async def proxy(store, chat, key, client_factory=httpx.AsyncClient):
    if chat.model != "miles-default":
        raise HTTPException(404, "Use model miles-default")
    deployment = store.active()
    if not deployment:
        raise HTTPException(503, "No model is deployed")
    request_id = reserve(store, key, deployment)
    payload = chat.model_dump(exclude_none=True)
    payload["model"] = "miles-qwen3-smoke" if deployment["endpoint"].endswith(":18080") else deployment["model_id"]
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    if chat.stream:
        payload["stream_options"] = {"include_usage": True}
    client = client_factory(timeout=httpx.Timeout(120, connect=5), trust_env=False)
    try:
        upstream = await client.send(
            client.build_request("POST", deployment["endpoint"] + "/v1/chat/completions", json=payload), stream=True
        )
    except (httpx.HTTPError, OSError):
        await client.aclose()
        record(store, request_id, 502, None)
        raise HTTPException(502, "Inference service is unavailable") from None
    if upstream.status_code != 200:
        status = upstream.status_code
        await upstream.aclose()
        await client.aclose()
        record(store, request_id, status, None)
        raise HTTPException(
            status if status < 500 else 502, "Inference request rejected; check context length and service health"
        )
    if not chat.stream:
        try:
            result = json.loads(await upstream.aread())
            result["model"] = "miles-default"
            record(store, request_id, 200, result.get("usage"))
            return JSONResponse(result, headers={"X-Request-ID": request_id})
        except (ValueError, httpx.HTTPError):
            record(store, request_id, 502, None)
            raise HTTPException(502, "Invalid inference response") from None
        finally:
            await upstream.aclose()
            await client.aclose()

    async def stream():
        usage, status, done = None, 200, False
        try:
            async for line in upstream.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    item = json.loads(line[6:])
                    item["model"] = "miles-default"
                    if item.get("usage"):
                        usage = item["usage"]
                    line = "data: " + json.dumps(item)
                if line == "data: [DONE]":
                    done = True
                yield (line + "\n").encode()
            if not done:
                status = 502
        except asyncio.CancelledError:
            status = 499
            raise
        except (ValueError, httpx.HTTPError):
            status = 502
            yield b'data: {"error":{"message":"Inference stream interrupted"}}\n\n'
        finally:
            record(store, request_id, status, usage)
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"}
    )
