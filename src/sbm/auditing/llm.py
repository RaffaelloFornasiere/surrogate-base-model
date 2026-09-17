"""Explicit LLM settings and recorded investigator/judge conversations."""

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    model: str
    temperature: float | None = None
    max_tokens: int = 400
    reasoning_effort: str | None = None


def make_client(*, base_url: str, api_key_env: str, timeout: float = 120, retries: int = 8):
    from openai import OpenAI

    return OpenAI(api_key=os.environ[api_key_env], base_url=base_url,
                  timeout=timeout, max_retries=retries)


def complete(client, messages: list[dict], config: LLMConfig) -> dict:
    kwargs = {"model": config.model, "max_tokens": config.max_tokens}
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.reasoning_effort is not None:
        kwargs["extra_body"] = {"reasoning_effort": config.reasoning_effort}
    response = client.chat.completions.create(messages=messages, **kwargs)
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice else None
    return {
        "model": config.model,
        "settings": asdict(config),
        "messages": messages,
        "output": (message.content or "") if message else "",
        "usage": response.usage.model_dump() if response.usage else {},
        "finish_reason": choice.finish_reason if choice else None,
        "refusal": getattr(message, "refusal", None),
    }


def parse_hypothesis(text: str) -> dict:
    quirk = re.search(r"<quirk>(.*?)</quirk>", text, re.S)
    description = re.search(r"<description>(.*?)</description>", text, re.S)
    return {
        "quirk": quirk.group(1).strip() if quirk else "",
        "description": description.group(1).strip() if description else text.strip(),
    }


def parse_verdict(text: str) -> dict:
    match = re.search(r"<match>\s*([01])\s*</match>", text)
    reason = re.search(r"<reason>(.*?)</reason>", text, re.S)
    return {
        "match": int(match.group(1)) if match else None,
        "reason": reason.group(1).strip() if reason else text.strip(),
    }


def investigate(client, prompt: str, config: LLMConfig, *, system_prompt: str = "") -> dict:
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    messages.append({"role": "user", "content": prompt})
    conversation = complete(client, messages, config)
    return {**parse_hypothesis(conversation["output"]), "investigator": conversation}


JUDGE_PROMPT = (Path(__file__).parent / "prompts" / "judge.txt").read_text().rstrip("\n")


def judge(client, identified: str, ground_truth: str, config: LLMConfig,
          *, template: str = JUDGE_PROMPT) -> dict:
    prompt = template.format(ground_truth=ground_truth, identified=identified)
    conversation = complete(client, [{"role": "user", "content": prompt}], config)
    parsed = parse_verdict(conversation["output"])
    return {**parsed, "raw": conversation["output"], "conversation": conversation}
