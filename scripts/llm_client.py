"""Call a model with a YAML spec plus free text, and return the reply.

Groq serves openai/gpt-oss-120b on its free tier. Gemini is available by
explicitly selecting a Gemini model. Calling OpenRouter is a separate switch
from allowing a paid model. Free mode sends a model only when that provider is
confirmed to serve it for free.
"""

import os
import re
import time
from pathlib import Path

import yaml

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_FREE_MODEL = "gemini-3.8-flash"
DEFAULT_MODEL = GROQ_MODEL
FREE_MODEL = "openrouter/free"  # $0 router; picks an available free model
# Tried in order by OpenRouter if an explicit paid model is rate-limited or down.
# route_models() omits them for a free request. The Groq fallback does not use them.
FALLBACK_MODELS = [
    "openai/gpt-5.6-terra",  # $2 / $12
    "deepseek/deepseek-v4-pro-0813",  # $0.58 / $1.73
]
_FALLBACK_ERRORS = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}
_FALLBACK_STATUS = {401, 403, 404, 408, 409, 429, 500, 502, 503, 529}


def is_free_model(model: str) -> bool:
    """True for an OpenRouter id that the provider marks as free."""
    return model == FREE_MODEL or model.endswith(":free")


def confirmed_free(provider: str, model: str) -> bool:
    """True only for a provider and model id already known to be free.

    openai/gpt-oss-120b is free on Groq. The same id on OpenRouter is billed,
    so it is not confirmed free there.
    """
    if provider == "groq" and model == GROQ_MODEL:
        return True
    if provider == "gemini" and model == GEMINI_FREE_MODEL:
        return True
    return provider == "openrouter" and is_free_model(model)


def _route_allowed(provider: str, model: str) -> bool:
    return confirmed_free(provider, model) or ALLOW_PAID


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# OpenRouter fallback stays available until a caller disables it.
# Paid fallback models stay off until a caller sets FACTTHREAD_ALLOW_PAID.
ALLOW_OPENROUTER_FALLBACK = _env_flag("FACTTHREAD_OPENROUTER_FALLBACK", True)
ALLOW_PAID = _env_flag("FACTTHREAD_ALLOW_PAID", False)


def configure_providers(*, allow_openrouter_fallback: bool | None = None, allow_paid: bool | None = None) -> None:
    """Set the provider switches for this process. Paid fallback stays off unless allow_paid is true."""
    global ALLOW_OPENROUTER_FALLBACK, ALLOW_PAID
    if allow_openrouter_fallback is not None:
        ALLOW_OPENROUTER_FALLBACK = allow_openrouter_fallback
    if allow_paid is not None:
        ALLOW_PAID = allow_paid


def route_models(model: str) -> list[str]:
    """OpenRouter model ids for one request.

    Free mode returns a confirmed-free id and nothing else. An unconfirmed id
    is omitted rather than sent as the primary model. Paid fallback ids are
    added only when paid models are allowed.
    """
    if not _route_allowed("openrouter", model):
        return []
    if not ALLOW_PAID:
        return [model]
    extras = [item for item in FALLBACK_MODELS if item != model]
    return [model, *extras]


def provider_chain(model: str) -> list[dict]:
    """Providers allowed for this model. OpenRouter and paid models are separate switches."""
    if model.startswith("gemini-"):
        if not _route_allowed("gemini", model):
            return []
        return [
            {
                "provider": "gemini",
                "model": model,
                "base_url": GEMINI_BASE_URL,
                "key_name": "GEMINI_API_KEY",
                "models": [model],
                "route_reason": "gemini_requested",
            }
        ]
    if model == GROQ_MODEL:
        chain = []
        if _route_allowed("groq", model):
            chain.append(
                {
                    "provider": "groq",
                    "model": GROQ_MODEL,
                    "base_url": GROQ_BASE_URL,
                    "key_name": "GROQ_API_KEY",
                    "models": [GROQ_MODEL],
                    "route_reason": "groq_primary",
                }
            )
        if ALLOW_OPENROUTER_FALLBACK and _route_allowed("openrouter", model):
            chain.append(
                {
                    "provider": "openrouter",
                    "model": GROQ_MODEL,
                    "base_url": OPENROUTER_BASE_URL,
                    "key_name": "OPENROUTER_API_KEY",
                    "models": [GROQ_MODEL],
                    "route_reason": "openrouter_same_model",
                }
            )
        return chain
    if not ALLOW_OPENROUTER_FALLBACK:
        return []
    models = route_models(model)
    if not models:
        return []
    return [
        {
            "provider": "openrouter",
            "model": models[0],
            "base_url": OPENROUTER_BASE_URL,
            "key_name": "OPENROUTER_API_KEY",
            "models": models,
            "route_reason": "openrouter_requested",
        }
    ]


# Teammates: replace "xxx" with your own OpenRouter key (starts with sk-or-).
# While it is still "xxx", the key is read from the environment, repo .env, or ~/.hermes/.env.
# Never commit a real key.
OPENROUTER_API_KEY = "xxx"


def _env_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [Path.cwd() / ".env", root / ".env", Path.home() / ".hermes" / ".env"]


def _read_key_from_env_file(path: Path, name: str) -> tuple[str, str]:
    """Return (key, status) where status is missing|empty|ok."""
    if not path.is_file():
        return "", "missing"
    found_blank = False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[7:].strip()
        found, _, value = s.partition("=")
        if found.strip() != name:
            continue
        key = value.strip().strip("\"'")
        if key and key != "xxx":
            return key, "ok"
        found_blank = True
    return "", "empty" if found_blank else "missing"


def lookup_key(name: str) -> tuple[str, str]:
    """Return (key, ok|empty|missing). Does not print the key."""
    if name == "OPENROUTER_API_KEY" and OPENROUTER_API_KEY and OPENROUTER_API_KEY != "xxx":
        return OPENROUTER_API_KEY, "ok"
    key = (os.environ.get(name) or "").strip()
    if key and key != "xxx":
        return key, "ok"
    saw_empty = False
    for path in _env_paths():
        key, status = _read_key_from_env_file(path, name)
        if status == "ok":
            return key, "ok"
        if status == "empty":
            saw_empty = True
    return "", "empty" if saw_empty else "missing"


def _load_api_key() -> str:
    """OpenRouter key: constant above, then process env, then local .env files."""
    key, status = lookup_key("OPENROUTER_API_KEY")
    if status == "ok":
        return key
    if status == "empty":
        raise RuntimeError(
            "Found .env but OPENROUTER_API_KEY is empty. Open .env in the project root "
            "and paste your key right after the equals sign, then restart the server."
        )
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Paste it into the gitignored .env next to README "
        "(OPENROUTER_API_KEY=sk-or-...) or export it in the same terminal as the server."
    )


def require_provider_key(model: str = DEFAULT_MODEL) -> None:
    """Raise when every provider for this model is missing a key. Does not call a model."""
    missing: list[str] = []
    for target in provider_chain(model):
        _, status = lookup_key(target["key_name"])
        if status == "ok":
            return
        missing.append(f"{target['key_name']} is {status}")
    raise RuntimeError(
        "No model key is configured (" + "; ".join(missing) + "). "
        "Set the API key for the selected provider (GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY)."
    )


def ask_llm(
    yaml_spec: str | dict,
    text: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    response_format: dict | None = None,
) -> str:
    """Send `yaml_spec` (instructions/schema) and `text` (user input) to the model.

    `yaml_spec` may be a YAML string or a dict (dumped to YAML). It goes in the
    system message; `text` goes in the user message. Returns the model's reply.
    """
    if isinstance(yaml_spec, dict):
        yaml_spec = yaml.safe_dump(yaml_spec, sort_keys=False, allow_unicode=True)
    else:
        yaml.safe_load(yaml_spec)  # fail early on malformed YAML

    response, _target = _complete(
        _messages(yaml_spec, text), model=model, temperature=temperature,
        include_usage=False, response_format=response_format,
    )
    return _message_text(response)


def _usage_dict(usage: object) -> dict:
    if usage is None:
        return {}
    data = usage.model_dump() if hasattr(usage, "model_dump") else {}
    extra = getattr(usage, "model_extra", None) or {}
    if isinstance(extra, dict):
        data = {**extra, **data}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        value = getattr(usage, name, None)
        if data.get(name) is None and value is not None:
            data[name] = value
    return data


def ask_llm_usage(
    yaml_spec: str | dict,
    text: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    response_format: dict | None = None,
    max_completion_tokens: int | None = None,
) -> dict:
    """Same call as ask_llm, plus the served model, token usage, cost, and elapsed seconds.

    cost_usd is "unknown" when the provider does not report a cost. It is not estimated and it is not recorded as zero.
    """
    if isinstance(yaml_spec, dict):
        yaml_spec = yaml.safe_dump(yaml_spec, sort_keys=False, allow_unicode=True)
    else:
        yaml.safe_load(yaml_spec)

    started = time.perf_counter()
    response, target = _complete(
        _messages(yaml_spec, text), model=model, temperature=temperature,
        include_usage=True, response_format=response_format,
        max_completion_tokens=max_completion_tokens,
    )
    elapsed = time.perf_counter() - started
    usage_data = _usage_dict(getattr(response, "usage", None))
    cost = usage_data.get("cost")
    reported = isinstance(cost, int | float)
    served = getattr(response, "model", None) or target["model"]
    attempts = target.get("attempts") or []
    if attempts:
        attempts[-1]["cost_usd"] = cost if reported else "unknown"
        attempts[-1]["served_model"] = served
    return {
        "content": _message_text(response),
        "model": served,
        "requested_model": target["model"],
        "provider": target["provider"],
        "route_reason": target.get("route_reason"),
        "http_status": target.get("http_status"),
        "cache_hit": False,
        "usage": usage_data,
        "rate_limit_headers": target.get("rate_limit_headers") or {},
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
        "max_completion_tokens": max_completion_tokens,
        "cost_usd": cost if reported else "unknown",
        "elapsed_s": elapsed,
        "attempts": attempts,
    }


def _messages(yaml_spec: str, text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Follow the YAML specification below when answering.\n\n" + yaml_spec},
        {"role": "user", "content": text},
    ]


def _message_text(response: object) -> str:
    message = response.choices[0].message
    content = message.content or ""
    if content.strip():
        return content
    extra = getattr(message, "model_extra", None) or {}
    return getattr(message, "reasoning", None) or (extra.get("reasoning") if isinstance(extra, dict) else "") or ""


def _client(target: dict):
    from openai import OpenAI

    # Retry policy is owned by the caller so rate limits and transient errors are recorded once.
    options = {"api_key": target["api_key"], "base_url": target["base_url"], "max_retries": 0}
    return OpenAI(**options)


class RateLimited(RuntimeError):
    """The provider refused the request. retry_after is seconds, when the server gave one."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = 429


def is_rate_limit(exc: BaseException) -> bool:
    if isinstance(exc, RateLimited):
        return True
    if getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError":
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text


def retry_after_seconds(exc: BaseException) -> float | None:
    """Read Retry-After from the error. A missing header stays missing."""
    direct = getattr(exc, "retry_after", None)
    if isinstance(direct, int | float):
        return float(direct)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw = headers.get("retry-after") if headers is not None and hasattr(headers, "get") else None
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str) and raw.strip().replace(".", "", 1).isdigit():
        return float(raw.strip())
    message = str(exc)
    minutes = re.search(r"(\d+)m(\d+(?:\.\d+)?)s", message)
    if minutes and is_rate_limit(exc):
        return int(minutes.group(1)) * 60 + float(minutes.group(2))
    seconds = re.search(r"(\d+(?:\.\d+)?)\s*s", message)
    if seconds and is_rate_limit(exc) and ("retry" in message.lower() or "try again" in message.lower()):
        return float(seconds.group(1))
    return None


def should_fallback(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    return status in _FALLBACK_STATUS or type(exc).__name__ in _FALLBACK_ERRORS


def _no_route_message(model: str) -> str:
    if not ALLOW_PAID and not any(
        confirmed_free(provider, model)
        for provider in ("groq", "gemini", "openrouter")
    ):
        return (
            f"{model} is not a confirmed free model. Paid models are not allowed, so it was not called. "
            "The fixed_rules baseline was not substituted."
        )
    if not ALLOW_OPENROUTER_FALLBACK and model != GROQ_MODEL and not model.startswith("gemini-"):
        return (
            f"OpenRouter is not allowed, so {model} was not called. "
            "The fixed_rules baseline was not substituted."
        )
    if ALLOW_OPENROUTER_FALLBACK and not ALLOW_PAID and model == GROQ_MODEL:
        return (
            "OpenRouter was not called because openai/gpt-oss-120b is not a confirmed free model there. "
            "Paid models are not allowed. The fixed_rules baseline was not substituted."
        )
    return f"No provider is allowed for {model}. The fixed_rules baseline was not substituted."


def available_targets(model: str) -> list[dict]:
    """Providers that have a key and are allowed for this model. Raises when none are."""
    chain = provider_chain(model)
    if not chain:
        raise RuntimeError(_no_route_message(model))
    ready: list[dict] = []
    missing: list[str] = []
    for target in chain:
        key, status = lookup_key(target["key_name"])
        if status == "ok":
            ready.append({**target, "api_key": key})
        else:
            missing.append(f"{target['key_name']} is {status}")
    if not ready:
        raise RuntimeError(
            "No model key is configured (" + "; ".join(missing) + "). "
            "Set the API key for the selected provider (GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY)."
        )
    return ready


def _attempt(target: dict) -> dict:
    return {
        "route_reason": target.get("route_reason"),
        "provider": target["provider"],
        "model": target["model"],
        "cache_hit": False,
        "ok": False,
        "cost_usd": "unknown",
    }


def _redact_provider_value(value):
    """Keep provider diagnostics while removing credentials and organization IDs."""
    if isinstance(value, dict):
        return {
            key: _redact_provider_value(item)
            for key, item in value.items()
            if str(key).lower() not in {"api_key", "authorization", "headers"}
        }
    if isinstance(value, list):
        return [_redact_provider_value(item) for item in value]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        value = re.sub(r"(?:gsk_|sk-)[A-Za-z0-9_-]+", "[REDACTED_KEY]", value)
        value = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED_KEY]", value)
        value = re.sub(r"org_[A-Za-z0-9]+", "[REDACTED_ORG]", value)
        value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._-]+", r"\1[REDACTED_KEY]", value)
    return value


def _provider_error_details(exc: BaseException) -> dict:
    body = getattr(exc, "body", None)
    response = getattr(exc, "response", None)
    if body is None and response is not None:
        body = getattr(response, "text", None)
        if body is None:
            content = getattr(response, "content", None)
            body = content if isinstance(content, bytes) else None
    safe_body = _redact_provider_value(body)
    error_object = body.get("error") if isinstance(body, dict) else None
    error_object = error_object if isinstance(error_object, dict) else {}
    request_parameters = getattr(exc, "factthread_request_parameters", None)
    return {
        "http_status": getattr(exc, "status_code", None),
        "exception_type": type(exc).__name__,
        "error_type": error_object.get("type") or getattr(exc, "type", None),
        "error_code": error_object.get("code") or getattr(exc, "code", None),
        "error_body": safe_body,
        "error_body_available": body is not None,
        "rate_limit_headers": _rate_limit_headers(getattr(response, "headers", None)),
        "request_parameters": _redact_provider_value(request_parameters),
    }


def _rate_limit_headers(headers: object) -> dict[str, str]:
    if headers is None or not hasattr(headers, "get"):
        return {}
    names = (
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    )
    return {name: str(headers.get(name)) for name in names if headers.get(name) is not None}


def _fail(message: str, exc: BaseException, attempts: list[dict]):
    if is_rate_limit(exc):
        error = RateLimited(message, retry_after=retry_after_seconds(exc))
    else:
        error = RuntimeError(message)
    error.attempts = attempts
    raise error from exc


def call_with_fallback(model: str, send):
    """Call send(client, target). A later provider is used only when that route is allowed."""
    targets = available_targets(model)
    errors: list[str] = []
    attempts: list[dict] = []
    for index, target in enumerate(targets):
        attempt = _attempt(target)
        client = _client(target)
        try:
            result = send(client, target)
        except Exception as exc:  # noqa: BLE001 — an allowed provider may follow a failed one
            attempt["error"] = type(exc).__name__
            attempt.update(_provider_error_details(exc))
            attempts.append(attempt)
            errors.append(f"{target['provider']} {target['model']}: {type(exc).__name__}")
            if index == len(targets) - 1 or not should_fallback(exc):
                message = "Model request failed. " + " ".join(errors) + " The fixed_rules baseline was not substituted."
                if ALLOW_OPENROUTER_FALLBACK and not ALLOW_PAID and model == GROQ_MODEL and target["provider"] == "groq":
                    message += " OpenRouter was not called because openai/gpt-oss-120b is not a confirmed free model there."
                _fail(message, exc, attempts)
            continue
        attempt["ok"] = True
        attempts.append(attempt)
        target["attempts"] = attempts
        return result, target
    raise RuntimeError("No model provider was called. The fixed_rules baseline was not substituted.")


def _complete(
    messages: list[dict[str, str]], model: str, temperature: float, include_usage: bool,
    response_format: dict | None = None,
    max_completion_tokens: int | None = None,
):
    def send(client, target):
        kwargs = {
            "model": target["model"],
            "temperature": temperature,
            "messages": messages,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        extra: dict = {}
        if target["provider"] == "openrouter":
            extra["models"] = target["models"]
            if include_usage:
                extra["usage"] = {"include": True}
        if extra:
            kwargs["extra_body"] = extra
        try:
            completions = client.chat.completions
            raw_api = getattr(completions, "with_raw_response", None)
            if raw_api is None:
                return completions.create(**kwargs)
            raw_response = raw_api.create(**kwargs)
            target["http_status"] = getattr(raw_response, "status_code", None)
            target["rate_limit_headers"] = _rate_limit_headers(getattr(raw_response, "headers", None))
            return raw_response.parse()
        except Exception as exc:
            try:
                exc.factthread_request_parameters = {
                    key: value for key, value in kwargs.items()
                    if key.lower() not in {"api_key", "authorization", "headers"}
                }
            except Exception:
                pass
            raise

    return call_with_fallback(model, send)


if __name__ == "__main__":
    spec = """
task: extract_bay_adjustment
output_format: json
fields: [bay, tier, row, action]
"""
    print(ask_llm(spec, "Move the container in bay 12 row 3 tier 2 to bay 14."))
