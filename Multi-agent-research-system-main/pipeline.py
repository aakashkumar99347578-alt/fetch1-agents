import sys
from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
    extract_text_content,
)

# Ensure console supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def run_research_pipeline(topic: str) -> dict:
    state = {"topic": topic}

    # Step 1 - Search Agent
    print("\n" + "=" * 50)
    print("Step 1 - Search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    raw_search = search_result['messages'][-1].content
    state["search_results"] = extract_text_content(raw_search)

    print("\nSearch Results:\n", state['search_results'])

    # Step 2 - Reader Agent
    print("\n" + "=" * 50)
    print("Step 2 - Reader agent is scraping top resources...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [(
            "user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    raw_scraped = reader_result['messages'][-1].content
    state['scraped_content'] = extract_text_content(raw_scraped)

    print("\nScraped Content:\n", state['scraped_content'])

    # Step 3 - Writer Chain
    print("\n" + "=" * 50)
    print("Step 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    report_result = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    state["report"] = extract_text_content(report_result)

    print("\nFinal Report:\n", state['report'])

    # Step 4 - Critic Review
    print("\n" + "=" * 50)
    print("Step 4 - Critic is reviewing the report...")
    print("=" * 50)

    feedback_result = critic_chain.invoke({
        "report": state['report']
    })
    state["feedback"] = extract_text_content(feedback_result)

    print("\nCritic Report:\n", state['feedback'])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    if topic.strip():
        run_research_pipeline(topic.strip())
    else:
        print("No topic entered. Exiting.")

