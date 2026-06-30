from pydantic_ai import Agent

from fit_kit.core.prompts import SLOT_PLANNER_PROMPT
from fit_kit.core.schemas import SlotPlan
from fit_kit.utils.log_utils import get_custom_logger

logger = get_custom_logger()


def create_planner_agent(
    model: str = "openai:gpt-5.5",
    retries: int = 2,
) -> Agent[None, SlotPlan]:
    """Factory function to create a planner agent that takes a natural language query and produces a SlotPlan:
    * Identifies which product categories are needed
    * Generates retrieval-optimized queries per slot
    * Extracts constraints (gender, season) and per-slot formality

    Uses pydantic-ai for structured output with OpenAI models.

    Example Usage:
    ```python
    from fit_kit.core.planner import create_planner

    planner = create_planner_agent("openai:gpt-5.5")

    result = planner.run_sync("outfit for a beach wedding this summer")
    plan = result.output  # SlotPlan
    ```

    :param model: pydantic-ai model string, e.g, "openai:gpt-4o", defaults to "openai:gpt-5.5"
    :param retries: number of retries on validation failure, defaults to 2
    :return: pydantic-ai Agent that outputs SlotPlan
    """
    return Agent(
        model,
        output_type=SlotPlan,
        system_prompt=SLOT_PLANNER_PROMPT,
        retries=retries,
    )


def plan(
    query: str,
    user_context: dict | None = None,
    model: str = "openai:gpt-5.5",
    retries: int = 2,
    agent: Agent[None, SlotPlan] | None = None,
) -> SlotPlan:
    """Function to plan an outfit from a natural language query.

    :param query: user's natural language request
    :param user_context: user profile and preferences, provided as context to the LLM.
            The LLM uses these as defaults but can override them based on the query
            (e.g. user is male but "outfit for my daughter" → gender=girls).
            Supported keys: "gender", "season", "max_price", "exclude",
            defaults to None
    :param model: pydantic-ai model string (ignored if agent is provided), defaults to "openai:gpt-5.5"
    :param retries: number of retries on validation failure, defaults to 2
    :param agent: reusable Agent instance (avoids re-creation per call), defaults to None
    :return: instance of SlotPlan
    """
    if agent is None:
        agent = create_planner_agent(model=model, retries=retries)

    user_msg = _build_user_message(query, user_context)

    result = agent.run_sync(user_msg)
    return result.output


async def aplan(
    query: str,
    user_context: dict | None = None,
    model: str = "openai:gpt-5.5",
    retries: int = 2,
    agent: Agent[None, SlotPlan] | None = None,
) -> SlotPlan:
    """Async function variant of plan() — yields control during the LLM API call.
    Plan an outfit from a natural language query.


    :param query: user's natural language request
    :param user_context: user profile and preferences, provided as context to the LLM.
            The LLM uses these as defaults but can override them based on the query
            (e.g. user is male but "outfit for my daughter" → gender=girls).
            Supported keys: "gender", "season", "max_price", "exclude",
            defaults to None
    :param model: pydantic-ai model string (ignored if agent is provided), defaults to "openai:gpt-5.5"
    :param retries: number of retries on validation failure, defaults to 2
    :param agent: reusable Agent instance (avoids re-creation per call), defaults to None
    :return: instance of SlotPlan
    """
    if agent is None:
        agent = create_planner_agent(model=model, retries=retries)
    user_msg = _build_user_message(query, user_context)
    result = await agent.run(user_msg)
    return result.output


def _build_user_message(query: str, user_context: dict | None) -> str:
    """Internal function to prepend user context to the query.

    Example:
    ```
    "[User: men, Preference: summer, Budget: $50] beach wedding outfit"
    ```

    Context is framed as user identity and preferences — the LLM treats
    them as defaults to follow unless the query clearly indicates otherwise.

    :param query: input query string
    :param user_context: optional dictionary of user context
    :return: updated query string with user context prepended
    """
    if not user_context:
        return query

    context_parts = []
    if "gender" in user_context:
        context_parts.append(f"User: {user_context['gender']}")
    if "season" in user_context:
        context_parts.append(f"Preference: {user_context['season']}")
    if "max_price" in user_context:
        context_parts.append(f"Budget: ${user_context['max_price']:.0f} per item")
    if "exclude" in user_context:
        context_parts.append(f"Avoid: {', '.join(user_context['exclude'])}")

    if not context_parts:
        return query

    return f"[{' | '.join(context_parts)}] {query}"
