from typing import Literal, TypedDict
import logging

class AgentState(TypedDict):
    query: str
    route: Literal["summary_tool", "search_tool", "action_tool"]
    result: str


def init_agents(tool_executer):
    tool = lambda state: tool_agent(state, tool_executer)
    action = action_agent
    merge = merge_agent

    return tool, action, merge

    
# Tool Agent that executes the query using the tool_executer
def tool_agent(state: AgentState, tool_executer) -> AgentState:
    query = state['query']
    response = tool_executer.query(query)

    # Get the tool name from the selector result
    selector_result = response.metadata.get('selector_result', None)
    if selector_result:
        try:
            selected_index = selector_result.selections[0].index
            tool_name = ["summary_tool", "search_tool", "action_tool"][selected_index]
        except Exception as e:
            logging.error(f"Error getting tool name: {e}")
            tool_name = 'search_tool'  # Default to search tool instead of unknown
    else:
        tool_name = 'search_tool'  # Default to search tool instead of unknown

    return {**state, 'result': str(response), 'route': tool_name}


# Action Agent that will take extra action with outside API
def action_agent(state: AgentState) -> AgentState:
    return state


# Merge Agent that validates the final output
def merge_agent(state: AgentState) -> AgentState:
    result = state.get('result', '').strip()

    if not result or result.lower() in ['n/a', 'none', 'no result found']:
        logging.warning("Merge agent: Empty or invalid result.")
        return {**state, 'result': "Sorry, I couldn't find a meaningful answer to your query."}

    logging.info("Merge agent: Result passed validation.")
    return state
