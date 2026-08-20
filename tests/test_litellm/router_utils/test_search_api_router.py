import logging
from types import SimpleNamespace

import pytest

from litellm.router_utils.search_api_router import SearchAPIRouter


class FakeSearchDispatcher:
    def __init__(self, failing_providers: frozenset[str] = frozenset()):
        self.calls: list[dict[str, object]] = []
        self.failing_providers = failing_providers

    async def __call__(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        if kwargs["search_provider"] in self.failing_providers:
            raise RuntimeError("provider unavailable")
        return "search response"


@pytest.mark.asyncio
async def test_search_tool_provider_params_are_forwarded_to_dispatcher():
    dispatcher = FakeSearchDispatcher()
    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "agentcore-search",
                "litellm_params": {
                    "search_provider": "agentcore",
                    "api_key": "configured-key",
                    "api_base": "https://configured.example.com/mcp",
                    "tool_name": "configured___WebSearch",
                },
            }
        ]
    )

    response = await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="agentcore-search",
        original_generic_function=dispatcher,
        query="LiteLLM",
    )

    assert response == "search response"
    assert dispatcher.calls[0]["tool_name"] == "configured___WebSearch"


@pytest.mark.asyncio
async def test_request_provider_params_override_search_tool_defaults():
    dispatcher = FakeSearchDispatcher()
    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "agentcore-search",
                "litellm_params": {
                    "search_provider": "agentcore",
                    "api_key": "configured-key",
                    "api_base": "https://configured.example.com/mcp",
                    "tool_name": "configured___WebSearch",
                },
            }
        ]
    )

    await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="agentcore-search",
        original_generic_function=dispatcher,
        query="LiteLLM",
        tool_name="request___WebSearch",
    )

    assert dispatcher.calls[0]["tool_name"] == "request___WebSearch"


@pytest.mark.asyncio
async def test_search_tool_credentials_are_forwarded_to_dispatcher():
    dispatcher = FakeSearchDispatcher()
    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "agentcore-search",
                "litellm_params": {
                    "search_provider": "agentcore",
                    "api_key": "configured-key",
                    "api_base": "https://configured.example.com/mcp",
                },
            }
        ]
    )

    await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="agentcore-search",
        original_generic_function=dispatcher,
        query="LiteLLM",
    )

    assert dispatcher.calls[0]["api_key"] == "configured-key"
    assert dispatcher.calls[0]["api_base"] == "https://configured.example.com/mcp"


@pytest.mark.asyncio
async def test_search_tool_credentials_follow_fallback_provider():
    dispatcher = FakeSearchDispatcher(failing_providers=frozenset({"primary"}))
    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "primary-search",
                "litellm_params": {
                    "search_provider": "primary",
                    "api_key": "primary-key",
                    "api_base": "https://primary.example.com",
                },
            },
            {
                "search_tool_name": "fallback-search",
                "litellm_params": {
                    "search_provider": "fallback",
                    "api_key": "fallback-key",
                    "api_base": "https://fallback.example.com",
                },
            },
        ]
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await SearchAPIRouter.async_search_with_fallbacks_helper(
            router_instance=router,
            model="primary-search",
            original_generic_function=dispatcher,
            query="LiteLLM",
        )

    response = await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="fallback-search",
        original_generic_function=dispatcher,
        query="LiteLLM",
    )

    assert response == "search response"
    assert [(call["search_provider"], call["api_key"], call["api_base"]) for call in dispatcher.calls] == [
        ("primary", "primary-key", "https://primary.example.com"),
        ("fallback", "fallback-key", "https://fallback.example.com"),
    ]


@pytest.mark.asyncio
async def test_search_tool_secrets_are_not_logged(caplog: pytest.LogCaptureFixture):
    dispatcher = FakeSearchDispatcher()
    api_key = "secret-configured-key"
    api_base = "https://secret-host.example.com/mcp"
    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "agentcore-search",
                "litellm_params": {
                    "search_provider": "agentcore",
                    "api_key": api_key,
                    "api_base": api_base,
                    "tool_name": "configured___WebSearch",
                },
            }
        ]
    )

    with caplog.at_level(logging.DEBUG):
        await SearchAPIRouter.async_search_with_fallbacks_helper(
            router_instance=router,
            model="agentcore-search",
            original_generic_function=dispatcher,
            query="LiteLLM",
        )

    assert api_key not in caplog.text
    assert api_base not in caplog.text
