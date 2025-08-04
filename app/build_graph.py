from langgraph.graph import StateGraph, END
from agents import init_agents, AgentState
from pipeline import LlamaIndexPipeline

def build_graph(data_dir):
    builder = StateGraph(AgentState)

    # Tool Agent Pipelines
    llama_pipeline = LlamaIndexPipeline()
    nodes = llama_pipeline.load_data(data_dir)
    llama_pipeline.build(nodes)
    tool_executer = llama_pipeline.tool_executer

    # Initiate Agents
    tool_agent, action_agent, merge_agent = init_agents(tool_executer)
    
    # Build graph
    builder.add_node("tool", tool_agent)
    builder.add_node("action", action_agent)
    builder.add_node("merge", merge_agent)

    builder.set_entry_point("tool")

    builder.add_conditional_edges(
        "tool",
        lambda state: state['route'],
        {
            "summary_tool": "merge",
            "search_tool": "merge",
            "action_tool": "action"  # only go to action state if needed
        }
    )

    builder.add_edge("action", "merge")
    builder.add_edge("merge", END)

    return builder.compile()
        

