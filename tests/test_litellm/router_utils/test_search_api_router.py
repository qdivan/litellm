from types import SimpleNamespace

import pytest

from litellm.router_utils.search_api_router import SearchAPIRouter


@pytest.mark.asyncio
async def test_search_tool_litellm_params_are_forwarded_to_selected_provider():
    captured_kwargs: dict[str, object] = {}

    async def search_provider(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "search response"

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
        original_generic_function=search_provider,
        query="LiteLLM",
    )

    assert response == "search response"
    assert captured_kwargs == {
        "search_provider": "agentcore",
        "api_key": "configured-key",
        "api_base": "https://configured.example.com/mcp",
        "tool_name": "configured___WebSearch",
        "query": "LiteLLM",
    }


@pytest.mark.asyncio
async def test_search_request_params_override_search_tool_litellm_params():
    captured_kwargs: dict[str, object] = {}

    async def search_provider(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "search response"

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
        original_generic_function=search_provider,
        query="LiteLLM",
        api_key="request-key",
        api_base="https://request.example.com/mcp",
        tool_name="request___WebSearch",
    )

    assert captured_kwargs["api_key"] == "request-key"
    assert captured_kwargs["api_base"] == "https://request.example.com/mcp"
    assert captured_kwargs["tool_name"] == "request___WebSearch"
