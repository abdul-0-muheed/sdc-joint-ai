import logging
import os
from google.genai import types
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    ChatContext,
    ChatMessage,
    function_tool,
    RunContext
)
from livekit.plugins import noise_cancellation, silero, google
from livekit.plugins.elevenlabs import TTS as ElevenLabsTTS
from livekit.plugins import assemblyai

# Import the new FAISS-based RAG system
from rag_faiss import rag_lookup

logger = logging.getLogger("agent")
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent")

# Function tool for RAG lookup with improved description
@function_tool()
async def rag_lookup_tool(context: RunContext, query: str):
    """Search for specific information about Shree Devi College in the database. 
    Use this tool for ANY question about Shree Devi College including:
    - Chairman, principal, faculty, or staff information
    - Courses, programs, or departments
    - Facilities, campus, or infrastructure
    - History, mission, or vision
    - Admissions, fees, or scholarships
    - Events, activities, or news
    
    Always use this tool first before answering any question about Shree Devi College."""
    logger.info(f"RAG lookup started for query: {query}")
    
    try:
        # Retrieve relevant college info using FAISS RAG
        rag_content = rag_lookup(query, top_k=5)
        logger.info(f"RAG lookup completed for query: {query}")
        
        if rag_content and rag_content.strip():
            # Log the RAG content that was found
            logger.info(f"RAG content found for query: {query}")
            logger.info(f"RAG content preview: n{rag_contet}")
            
            # Return formatted content
            return f"Information from Shree Devi College database:\n{rag_content}"
        else:
            logger.info(f"No RAG content found for query: {query}")
            return "No specific information found in Shree Devi College database for this query."
    except Exception as e:
        logger.error(f"Error in RAG lookup for query '{query}': {str(e)}")
        return f"An error occurred while searching the database: {str(e)}"

# Custom web search tool as a function tool
@function_tool()
async def web_search_tool(context: RunContext, query: str):
    """Search the web for information about Shree Devi College when it's not available in the database.
    Use this tool only when the rag_lookup_tool doesn't provide sufficient information."""
    logger.info(f"Web search started for query: {query}")
    
    try:
        # In a real implementation, you would use a web search API here
        # For now, we'll return a placeholder message
        logger.info(f"Web search completed for query: {query}")
        return f"Web search results for '{query}': [This would contain web search results in a real implementation]"
    except Exception as e:
        logger.error(f"Error in web search for query '{query}': {str(e)}")
        return f"An error occurred while searching the web: {str(e)}"

# Simple tool that doesn't need RAG context
@function_tool()
async def my_tool(context: RunContext):
    """A simple tool that responds with 'Good boy'. Use this when the user specifically asks to use the tool."""
    logger.info("my_tool called")
    await context.session.say("Good boy")

# # Function tool for external information (when not in database)
# @function_tool()
# async def external_info_tool(context: RunContext, query: str):
#     """Search for information from external sources when not available in the college database. Use this when the rag_lookup_tool doesn't provide sufficient information."""
#     logger.info(f"External info lookup started for query: {query}")
    
#     try:
#         # Use the Google search capability that's already available
#         # Note: This is a placeholder - in a real implementation, you might use an API call
#         # For now, we'll rely on the LLM's built-in search capability
        
#         logger.info(f"External info lookup completed for query: {query}")
#         return "Please use web search to find this information as it's not available in the college database."
#     except Exception as e:
#         logger.error(f"Error in external info lookup for query '{query}': {str(e)}")
#         return f"An error occurred while searching for external information: {str(e)}"    

# ----------------------
# Assistant class
# ----------------------
class CollegeAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=("""You are a helpful AI assistant for Shree Devi College.
            You only answer questions related to Shree Devi College.
            If asked about other colleges or topics, politely say you only provide info about Shree Devi College.
            
            IMPORTANT: For ANY question about Shree Devi College, you MUST first use the rag_lookup_tool to search for information in the database.
            This includes questions about:
            - Chairman, principal, faculty, or staff
            - Courses, programs, or departments
            - Facilities, campus, or infrastructure
            - History, mission, or vision
            - Admissions, fees, or scholarships
            - Events, activities, or news
            
            After receiving the tool result, formulate a comprehensive answer using that information.
            If the rag_lookup_tool doesn't provide sufficient information, use the web_search_tool to search for external information.
            Always cite your sources when using external information.
            Prioritize accuracy and relevance to Shree Devi College.
            
            If the user says 'tool', use the my_tool function."""),
            tools=[rag_lookup_tool, web_search_tool, my_tool],
        )
    
    # Add logging for when the agent is about to generate a response
    async def will_llm_generate(self, turn_ctx: ChatContext):
        logger.info("LLM is about to generate a response")
        # Log the current chat context for debugging
        for i, msg in enumerate(turn_ctx.messages):
            if hasattr(msg, 'content'):
                content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                logger.info(f"Message {i+1} - Role: {msg.role}, Content: {content_preview}")
            else:
                logger.info(f"Message {i+1} - Role: {msg.role}, Content: [No content]")

# ----------------------
# Prewarm function
# ----------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# ----------------------
# Entry point
# ----------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Starting session for room: {ctx.room.name}")

    session = AgentSession(
        stt = assemblyai.STT(),
        llm=google.LLM(
            model="gemini-2.5-flash",
            temperature=0.5,  # Lower temperature for more deterministic behavior
            # Removed gemini_tools to avoid mixing tool types
        ),
        tts=ElevenLabsTTS(
        model="eleven_multilingual_v2",
        voice_id="ODq5zmih8GrVes37Dizd",
    ),
    #      tts=google.beta.GeminiTTS(
    #     model="gemini-2.5-flash-preview-tts",
    #     voice_name="Zephyr",
    #     instructions="Speak in a friendly and engaging tone.",
    # ),
    # llm=google.realtime.RealtimeModel(
    #     model="gemini-2.0-flash-exp",
    #     voice="Puck",
    #     temperature=0.8,
    #     instructions="You are a helpful assistant",
    # ),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=CollegeAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    logger.info("Session started, generating initial greeting")
    await session.generate_reply(
        instructions="Greet the user and offer help about Shree Devi College."
    )
    await ctx.connect()

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))




#########################


import logging
import os
from google.genai import types
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    ChatContext,
    ChatMessage,
    function_tool,
    RunContext
)
from livekit.plugins import noise_cancellation, silero, google
from livekit.plugins.elevenlabs import TTS as ElevenLabsTTS
from livekit.plugins import assemblyai

# Import the new FAISS-based RAG system
from rag_faiss import rag_lookup

logger = logging.getLogger("agent")
load_dotenv()

# Configure logging - reduced for performance
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING to reduce log overhead
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Function tool for RAG lookup with improved description
@function_tool()
async def rag_lookup_tool(context: RunContext, query: str):
    """Search for specific information about Shree Devi College in the database. 
    Use this tool for ANY question about Shree Devi College including:
    - Chairman, principal, faculty, or staff information
    - Courses, programs, or departments
    - Facilities, campus, or infrastructure
    - History, mission, or vision
    - Admissions, fees, or scholarships
    - Events, activities, or news
    
    Always use this tool first before answering any question about Shree Devi College."""
    logger.debug(f"RAG lookup started for query: {query}")
    
    try:
        # Increased top_k to get more comprehensive results
        rag_content = rag_lookup(query, top_k=20)  # Increased to get more results
        logger.debug(f"RAG lookup completed for query: {query}")
        
        if rag_content :
            logger.debug(f"RAG content found for query: {rag_content}")
            return f"Information from Shree Devi College database:\n{rag_content}"
        else:
            logger.debug(f"No RAG content found for query: {query}")
            return "No specific information found in Shree Devi College database for this query."
    except Exception as e:
        logger.error(f"Error in RAG lookup for query '{query}': {str(e)}")
        return f"An error occurred while searching the database: {str(e)}"

# Simple tool that doesn't need RAG context
@function_tool()
async def my_tool(context: RunContext):
    """A simple tool that responds with 'Good boy'. Use this when the user specifically asks to use the tool."""
    logger.debug("my_tool called")
    await context.session.say("Good boy")

# ----------------------
# Assistant class
# ----------------------
class CollegeAssistant(Agent):
    def __init__(self) -> None:
        
        super().__init__(
            instructions=("""You are a helpful AI assistant for Shree Devi College.
            You only answer questions related to Shree Devi College.
            If asked about other colleges or topics, politely say you only provide info about Shree Devi College.
            
            IMPORTANT: For ANY question about Shree Devi College, you MUST first use the rag_lookup_tool to search for information in the database.
            
            donot say i dont have data make a made up answer based on the data you have and given 
            After receiving the tool result, formulate a detailed answer from the llm dataset answer using that information.
            
            
            
            
            If the user says 'tool', use the my_tool function."""),
            tools=[rag_lookup_tool, my_tool],
        )
    
    # Reduced logging for performance
    async def will_llm_generate(self, turn_ctx: ChatContext):
        logger.debug("LLM is about to generate a response")
        # Only log the last message to reduce overhead
        if turn_ctx.messages and hasattr(turn_ctx.messages[-1], 'content'):
            last_msg = turn_ctx.messages[-1]
            content_preview = last_msg.content[:50] + "..." if len(last_msg.content) > 50 else last_msg.content
            logger.debug(f"Last message - Role: {last_msg.role}, Content: {content_preview}")

    async def on_user_turn_completed(
            self, turn_ctx: ChatContext, new_message: ChatMessage,
        ) -> None:
            """Inject context for general questions automatically."""
            if new_message.role != "user":
                return
                
            query = new_message.text_content()
            
            # Only inject context for general questions
            if self._is_general_question(query):
                logger.debug(f"Injecting context for general question: {query}")
                rag_content = rag_lookup(query, top_k=3)
                turn_ctx.add_message(
                    role="system", 
                    content=f"General information about Shree Devi College:\n{rag_content}"
                )

# ----------------------
# Prewarm function
# ----------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# ----------------------
# Entry point
# ----------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Starting session for room: {ctx.room.name}")

    session = AgentSession(
        stt = assemblyai.STT(),
        llm=google.LLM(
            model="gemini-2.5-flash",  # Changed from flash-lite to full flash for better context processing
            temperature=0.5,
            # Removed gemini_tools to avoid mixing tool types
        ),
        tts=ElevenLabsTTS(
        model="eleven_multilingual_v2",
        voice_id="ODq5zmih8GrVes37Dizd",
    ),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=CollegeAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    logger.info("Session started, generating initial greeting")
    await session.generate_reply(
        instructions="Greet the user and offer help about Shree Devi College."
    )
    await ctx.connect()

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
