import asyncio
import base64
import json
from typing import Generic, Optional, TypeVar, Dict, Any, List
import os
import asyncio.exceptions

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import BaseTool, ArgSchema
from omagent_core.models.llms.base import BaseLLM
from omagent_core.utils.logger import logging
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM

# Default timeout for browser operations (in seconds)
DEFAULT_TIMEOUT = 30

Context = TypeVar("Context")

ARGSCHEMA = {
    "action": {
        "type": "string",
        "enum": [
            "go_to_url",
            "click_element",
            "input_text",
            "scroll_down",
            "scroll_up",
            "scroll_to_text",
            "send_keys",
            "get_dropdown_options",
            "select_dropdown_option",
            "go_back",
            "wait",
            "extract_content",
            "switch_tab",
            "open_tab",
            "close_tab",
            "web_search",
        ],
        "description": "The browser action to perform",
        "required": True,
    },
    "url": {
        "type": "string",
        "description": "URL for 'go_to_url' or 'open_tab' actions",
    },
    "index": {
        "type": "integer",
        "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
    },
    "text": {
        "type": "string",
        "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
    },
    "scroll_amount": {
        "type": "integer",
        "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
    },
    "tab_id": {
        "type": "integer",
        "description": "Tab ID for 'switch_tab' action",
    },
    "goal": {
        "type": "string",
        "description": "Extraction goal for 'extract_content' action",
    },
    "keys": {
        "type": "string",
        "description": "Keys to send for 'send_keys' action",
    },
    "seconds": {
        "type": "integer",
        "description": "Seconds to wait for 'wait' action",
    },
    "query": {
        "type": "string",
        "description": "Search query for 'web_search' action",
    },
}


@registry.register_tool()
class BrowserUseTool(BaseTool, Generic[Context]):
    """A powerful browser automation tool that allows interaction with web pages through various actions."""

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

    class Config:
        """Configuration for this pydantic object."""
        extra = "allow"
        arbitrary_types_allowed = True

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    
    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    # LLM for content extraction
    llm: OpenaiGPTLLM = {
        "name": "OpenaiGPTLLM", 
        "model_id": "ep-20250214102831-7mfjh",  
        "api_key": os.getenv("custom_openai_key"), 
        "endpoint": "https://ark.cn-beijing.volces.com/api/v3",
        "vision": False,
        "response_format": {"type": "text"},
        "use_default_sys_prompt": False,
        "temperature": 0.01,
        "max_tokens": 4096,
    }
    web_search_tool: Any = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        
        # Initialize LLM if not already a concrete instance
        if isinstance(self.llm, dict):
            from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
            self.llm = OpenaiGPTLLM(**self.llm)
        elif self.llm is None:
            from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
            self.llm = OpenaiGPTLLM(
                model_id="gpt-3.5-turbo",
                temperature=0.01,
                max_tokens=4096
            )

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}
            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()
            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def _arun(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a specified browser action asynchronously.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for web search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            Dict with the action's output and other relevant information
        """
        # Create a timeout for all browser operations
        timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)
        
        async def _execute_with_timeout():
            async with self.lock:
                try:
                    context = await self._ensure_browser_initialized()

                    # Default max content length
                    max_content_length = 2000

                    # Navigation actions
                    if action == "go_to_url":
                        if not url:
                            return {"error": "URL is required for 'go_to_url' action"}
                        page = await context.get_current_page()
                        await page.goto(url)
                        await page.wait_for_load_state()
                        return {"output": f"Navigated to {url}"}

                    elif action == "go_back":
                        await context.go_back()
                        return {"output": "Navigated back"}

                    elif action == "refresh":
                        await context.refresh_page()
                        return {"output": "Refreshed current page"}
                    
                    elif action == "web_search":
                        if not query:
                            return {"error": "Query is required for 'web_search' action"}
                        
                        # If web_search_tool is not initialized, initialize it
                        if self.web_search_tool is None:
                            from omagent_core.tool_system.tools.web_search.search import WebSearch
                            self.web_search_tool = WebSearch()
                        
                        # Execute the web search
                        search_response = await self.web_search_tool.arun(
                            {"search_query": query, "num_results": 1, "region": "en-US"}
                        )
                        
                        # Navigate to the first search result if available
                        if search_response and len(search_response) > 0:
                            first_search_result = search_response[0]
                            url_to_navigate = first_search_result.get("url", "")
                            
                            if url_to_navigate:
                                page = await context.get_current_page()
                                await page.goto(url_to_navigate)
                                await page.wait_for_load_state()

                        return {"output": f"Performed web search for '{query}' and navigated to the first result", "results": search_response}

                    # Element interaction actions
                    elif action == "click_element":
                        if index is None:
                            return {"error": "Index is required for 'click_element' action"}
                        element = await context.get_dom_element_by_index(index)
                        if not element:
                            return {"error": f"Element with index {index} not found"}
                        download_path = await context._click_element_node(element)
                        output = f"Clicked element at index {index}"
                        if download_path:
                            output += f" - Downloaded file to {download_path}"
                        return {"output": output}

                    elif action == "input_text":
                        if index is None or not text:
                            return {"error": "Index and text are required for 'input_text' action"}
                        element = await context.get_dom_element_by_index(index)
                        if not element:
                            return {"error": f"Element with index {index} not found"}
                        await context._input_text_element_node(element, text)
                        return {"output": f"Input '{text}' into element at index {index}"}

                    elif action == "scroll_down" or action == "scroll_up":
                        direction = 1 if action == "scroll_down" else -1
                        amount = (
                            scroll_amount
                            if scroll_amount is not None
                            else context.config.browser_window_size["height"]
                        )
                        await context.execute_javascript(
                            f"window.scrollBy(0, {direction * amount});"
                        )
                        return {"output": f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"}

                    elif action == "scroll_to_text":
                        if not text:
                            return {"error": "Text is required for 'scroll_to_text' action"}
                        page = await context.get_current_page()
                        try:
                            locator = page.get_by_text(text, exact=False)
                            await locator.scroll_into_view_if_needed()
                            return {"output": f"Scrolled to text: '{text}'"}
                        except Exception as e:
                            return {"error": f"Failed to scroll to text: {str(e)}"}

                    elif action == "send_keys":
                        if not keys:
                            return {"error": "Keys are required for 'send_keys' action"}
                        page = await context.get_current_page()
                        await page.keyboard.press(keys)
                        return {"output": f"Sent keys: {keys}"}

                    elif action == "get_dropdown_options":
                        if index is None:
                            return {"error": "Index is required for 'get_dropdown_options' action"}
                        element = await context.get_dom_element_by_index(index)
                        if not element:
                            return {"error": f"Element with index {index} not found"}
                        page = await context.get_current_page()
                        options = await page.evaluate(
                            """
                            (xpath) => {
                                const select = document.evaluate(xpath, document, null,
                                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (!select) return null;
                                return Array.from(select.options).map(opt => ({
                                    text: opt.text,
                                    value: opt.value,
                                    index: opt.index
                                }));
                            }
                        """,
                            element.xpath,
                        )
                        return {"output": f"Dropdown options: {options}", "options": options}

                    elif action == "select_dropdown_option":
                        if index is None or not text:
                            return {"error": "Index and text are required for 'select_dropdown_option' action"}
                        element = await context.get_dom_element_by_index(index)
                        if not element:
                            return {"error": f"Element with index {index} not found"}
                        page = await context.get_current_page()
                        await page.select_option(element.xpath, label=text)
                        return {"output": f"Selected option '{text}' from dropdown at index {index}"}

                    # Content extraction actions
                    elif action == "extract_content":
                        if not goal:
                            return {"error": "Goal is required for 'extract_content' action"}

                        page = await context.get_current_page()
                        import markdownify

                        content = markdownify.markdownify(await page.content())

                        prompt = f"""\
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.
Extraction goal: {goal}

Page content:
{content[:max_content_length]}
"""
                        messages = [{"role": "system", "content": prompt}]

                        # Define extraction function schema
                        extraction_function = {
                            "type": "function",
                            "function": {
                                "name": "extract_content",
                                "description": "Extract specific information from a webpage based on a goal",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "extracted_content": {
                                            "type": "object",
                                            "description": "The content extracted from the page according to the goal",
                                            "properties": {
                                                "text": {
                                                    "type": "string",
                                                    "description": "Text content extracted from the page",
                                                },
                                                "metadata": {
                                                    "type": "object",
                                                    "description": "Additional metadata about the extracted content",
                                                    "properties": {
                                                        "source": {
                                                            "type": "string",
                                                            "description": "Source of the extracted content",
                                                        }
                                                    },
                                                },
                                            },
                                        }
                                    },
                                    "required": ["extracted_content"],
                                },
                            },
                        }

                        # Use LLM to extract content with required function calling
                        response = await self.llm.ask_tool(
                            messages,
                            tools=[extraction_function],
                            tool_choice="required",
                        )

                        extracted_content = {}
                        if response and hasattr(response, "tool_calls") and response.tool_calls:
                            args = json.loads(response.tool_calls[0].function.arguments)
                            extracted_content = args.get("extracted_content", {})
                        
                        return {"output": f"Extracted from page", "extracted_content": extracted_content}

                    # Tab management actions
                    elif action == "switch_tab":
                        if tab_id is None:
                            return {"error": "Tab ID is required for 'switch_tab' action"}
                        await context.switch_to_tab(tab_id)
                        page = await context.get_current_page()
                        await page.wait_for_load_state()
                        return {"output": f"Switched to tab {tab_id}"}

                    elif action == "open_tab":
                        if not url:
                            return {"error": "URL is required for 'open_tab' action"}
                        await context.create_new_tab(url)
                        return {"output": f"Opened new tab with {url}"}

                    elif action == "close_tab":
                        await context.close_current_tab()
                        return {"output": "Closed current tab"}

                    # Utility actions
                    elif action == "wait":
                        seconds_to_wait = seconds if seconds is not None else 3
                        # Limit wait time to prevent long hanging
                        seconds_to_wait = min(seconds_to_wait, 30)
                        await asyncio.sleep(seconds_to_wait)
                        return {"output": f"Waited for {seconds_to_wait} seconds"}

                    else:
                        return {"error": f"Unknown action: {action}"}

                except asyncio.TimeoutError:
                    # Handle timeout errors
                    return {"error": f"Browser action '{action}' timed out after {timeout} seconds"}
                except Exception as e:
                    print(f"Browser action '{action}' failed: {e}")
                    return {"error": f"Browser action '{action}' failed: {str(e)}"}
        
        try:
            return await asyncio.wait_for(_execute_with_timeout(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Browser action '{action}' timed out")
            # Force cleanup on timeout
            await self.cleanup()
            return {"error": f"Browser action '{action}' timed out after {timeout} seconds"}
        except Exception as e:
            print(f"Unexpected error during browser action: {e}")
            # Ensure cleanup on any error
            await self.cleanup()
            return {"error": f"Unexpected error during browser action: {str(e)}"}

    def _run(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a specified browser action synchronously.
        This runs the async version in a new event loop.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(
            self._arun(
                action=action,
                url=url,
                index=index,
                text=text,
                scroll_amount=scroll_amount,
                tab_id=tab_id,
                query=query,
                goal=goal,
                keys=keys,
                seconds=seconds,
                **kwargs,
            )
        )
        return result

    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current browser state.
        Returns a dictionary with the state information.
        """
        try:
            if not self.context:
                return {"error": "Browser context not initialized"}

            state = await self.context.get_state()

            # Default viewport height if not available
            viewport_height = 800
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(self.context, "config") and hasattr(self.context.config, "browser_window_size"):
                viewport_height = self.context.config.browser_window_size.get("height", 800)

            # Take a screenshot for the state
            page = await self.context.get_current_page()

            await page.bring_to_front()
            await page.wait_for_load_state()

            screenshot = await page.screenshot(
                full_page=True, animations="disabled", type="jpeg", quality=100
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return {
                "output": json.dumps(state_info, indent=4, ensure_ascii=False),
                "base64_image": screenshot,
            }
        except Exception as e:
            print(f"Failed to get browser state: {e}")
            return {"error": f"Failed to get browser state: {str(e)}"}

    async def cleanup(self):
        """Clean up browser resources."""
        try:
            async with self.lock:
                if self.context is not None:
                    try:
                        await asyncio.wait_for(self.context.close(), timeout=5)
                    except (asyncio.TimeoutError, Exception) as e:
                        print(f"Error closing browser context: {e}")
                    finally:
                        self.context = None
                        self.dom_service = None
                
                if self.browser is not None:
                    try:
                        await asyncio.wait_for(self.browser.close(), timeout=5)
                    except (asyncio.TimeoutError, Exception) as e:
                        print(f"Error closing browser: {e}")
                    finally:
                        self.browser = None
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
                else:
                    loop.run_until_complete(self.cleanup())
            except Exception as e:
                print(f"Error during __del__ cleanup: {e}")
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.cleanup())
                    loop.close()
                except Exception:
                    pass

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool 