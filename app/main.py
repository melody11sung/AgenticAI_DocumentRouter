from build_graph import build_graph
import logging
from dotenv import load_dotenv
import time

# Load openai api key
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s <%(name)s> %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_workflow():

    data_dir = "docs"
    graph = build_graph(data_dir)
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    test_cases = [
        {
            "name": "Out of scope query",
            "query": "Where is Georgia Tech located?",
            "expected_route": "action_tool"
        },
        {
            "name": "Basic Summary",
            "query": "What is the main idea of the document?",
            "expected_route": "summary_tool"
        },
        {
            "name": "Search for specific information",
            "query": "How many dataset did author Aaqib used in his paper?",
            "expected_route": "search_tool"
        },
        {
            "name": "Out of scope query",
            "query": "How do you treat a broken bone?",
            "expected_route": "action_tool"
        }
    ]
    results = []

    for case in test_cases:
        start_time = time.time()
        state = {"query": case["query"], "route": "", "result": ""}
        
        try:
            output = graph.invoke(state)
            elapsed = time.time() - start_time
            
            result = {
                "name": case["name"],
                "query": case["query"],
                "route": output["route"],
                "expected_route": case["expected_route"],
                "result": output["result"],
                "elapsed_time": round(elapsed, 2),
                "pass": output["route"] == case["expected_route"]
            }
            results.append(result)
        except Exception as e:
            logger.error(f"Error in test {case['name']}: {e}")
            results.append({
                "name": case["name"],
                "query": case["query"],
                "route": "error",
                "expected_route": case["expected_route"],
                "result": str(e),
                "elapsed_time": round(elapsed, 2),
                "pass": False
            })

    logger.info("===== Workflow Completed =====")
    for res in results:
        print(f"\nTest: {res['name']}")
        print(f"Query: {res['query']}")
        print(f"Result: {res['result']}")
        print(f"Pass: {res['pass']}, Expected Route: {res['expected_route']}, Actual Route: {res['route']}")
        print(f"Elapsed Time: {res['elapsed_time']} seconds")


if __name__ == "__main__":
    evaluate_workflow()