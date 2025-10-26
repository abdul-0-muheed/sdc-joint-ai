import logging
import os
import asyncio
import multiprocessing
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    ChatContext,
    ChatMessage,
    function_tool,
    RunContext,
    BackgroundAudioPlayer,
    AudioConfig,
    BuiltinAudioClip,
    inference,
    RunContext
)
from livekit.plugins import noise_cancellation, silero, google
from livekit.plugins.elevenlabs import TTS as ElevenLabsTTS
from livekit.plugins import assemblyai
from supabase import create_client, Client
from typing import Optional, Dict, Any
from livekit.rtc import Room
import json


# Assuming async_rag_lookup is still relevant for other tools or fallback
# from rag_faiss import async_rag_lookup 

logger = logging.getLogger("agent")
load_dotenv()


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Initialize Supabase client
def get_supabase_client() -> Client:
    """Initialize and return Supabase client"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SECRET_KEY")   
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    return create_client(supabase_url, supabase_key)


Room = None

# @function_tool()
# async def rag_lookup_tool(context: RunContext, query: str):
#     """Search for specific information about Shree Devi College in the database.
#     Use this tool for complex questions or follow-up searches about:
#     - Chairman, principal, faculty, or staff information
#     - Courses, programs, or departments
#     - Facilities, campus, or infrastructure
#     - History, mission, or vision
#     - Admissions, fees, or scholarships
#     - Events, activities, or news
#     """
#     logger.debug(f"RAG lookup tool called for query: {query}")

#     async def _speak_status_update(delay: float = 1.0):
#         await asyncio.sleep(delay)
#         if not lookup_task.done():
#             await context.session.generate_reply(
#                 instructions=f"Searching the Shree Devi College database for '{query}'. This may take a moment."
#             )

#     status_update_task = asyncio.create_task(_speak_status_update(delay=1.0))
#     lookup_task = asyncio.create_task(async_rag_lookup(query, top_k=10))

#     try:
#         rag_content = await lookup_task
#     except Exception as e:
#         logger.error(f"Error in RAG lookup tool for query '{query}': {str(e)}")
#         rag_content = f"An error occurred while searching the database: {str(e)}"

#     status_update_task.cancel()
#     try:
#         await status_update_task
#     except asyncio.CancelledError:
#         pass

#     logger.debug(f"RAG lookup tool completed for query: {query}")
#     return rag_content
# Add these functions before the CollegeAssistant class definition
@function_tool()
async def contact_helpdesk(
    context: RunContext,
    requester_name: str,
    requester_email: str,
    phone_number: str,
    request_text: str
) -> str:
    """
    Submits a helpdesk request to the Supabase database.
    
    Args:
        context (RunContext): The LiveKit context object
        requester_name (str): Name of the person submitting the request
        requester_email (str): Email of the person submitting the request
        phone_number (str): Phone number of the person submitting the request
        request_text (str): The actual request text
        
    Returns:
        str: Confirmation message with request details or error message
    """
    print(f"DEBUG: contact_helpdesk called with name='{requester_name}', email='{requester_email}', phone='{phone_number}'")
    
    try:
        # Initialize Supabase client
        supabase = get_supabase_client()
        print("DEBUG: Supabase client initialized")
        
        logger.info(f"Submitting helpdesk request for: {requester_name}")
        
        # Prepare the data
        request_data = {
            "requester_name": requester_name,
            "requester_email": requester_email,
            "phone_number": phone_number,
            "request_text": request_text,
            "status": "Pending"
        }
        
        # Insert the request
        response = supabase.table("helpdesk_requests").insert([request_data]).execute()
        print(f"DEBUG: Supabase insert completed. Response type: {type(response)}")
        print(f"DEBUG: Response data: {response.data}")
        
        # Check for errors in the response
        if hasattr(response, 'error') and response.error:
            raise Exception(response.error.message)
        
        if not response.data:
            logger.error("Failed to submit helpdesk request - no data returned")
            return "❌ Failed to submit your helpdesk request. No data was returned from the server."
        
        # Return success message
        request_id = response.data[0]['id']
        success_message = f"✅ Your helpdesk request has been submitted successfully. Your request ID is {request_id}. We will review it and get back to you soon."
        logger.info(f"Helpdesk request submitted successfully with ID: {request_id}")
        return success_message
        
    except Exception as e:
        print(f"DEBUG: Exception in contact_helpdesk: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error submitting helpdesk request: {str(e)}", exc_info=True)
        
        # Check if the error is related to RLS policies
        if "row-level security policy" in str(e):
            return "❌ There's a permission issue with the helpdesk system. The administrator needs to update the security policies for the helpdesk system. Please try again later or contact support."
        
        return f"❌ Error submitting helpdesk request: {str(e)}"

@function_tool()
async def manual_callback(context: RunContext) -> str:
    """
    Handles manual callback request submission by collecting user information and request details.
    This function guides the user through the the link to fill.
    """



    try:
        global Room
        message_data = {
            "headline": "Submit a callback request",
            "text": "Please fill out the form below to get a call from collage.",
            "link": "/StudentAssistance/HelpdeskRequeste",
            "linkText": "Go to callback Form"
        }
        import json
        message_text = json.dumps(message_data)

        # ✅ Correct access pattern:
        # In AgentSession, you must connect manually if you want the rtc.Room instance.
        

        # ✅ Send message to frontend
        info = await Room.local_participant.send_text(
            message_text,
            topic="cloud-message"
        )
        logger.info(f"Sent text with stream ID: {info.stream_id}")

        # ✅ Also speak response
        await context.session.say(
            "I've opened the complaint form for you. Please fill in your details and submit your complaint."
        )

        return "Complaint form opened successfully."

    except Exception as e:
        logger.error(f"Error in manual_complaint: {str(e)}")
        await context.session.say(
            "Please visit the complaint section in the Student Assistance menu to submit your complaint."
        )
        return f"Sorry, I encountered an error: {str(e)}"



@function_tool()
async def submit_complaint(
    context: RunContext,
    student_name: str,
    student_email: str,
    department: str,
    complaint_text: str
) -> str:
    """
    Submits a student complaint to the Supabase database using the same approach as the frontend.
    
    Args:
        context (RunContext): The LiveKit context object
        student_name (str): Name of the student submitting the complaint
        student_email (str): Email of the student submitting the complaint
        department (str): Department of the student
        complaint_text (str): The actual complaint text
        
    Returns:
        str: Confirmation message with complaint details or error message and say to user
    """
    print(f"DEBUG: submit_complaint called with name='{student_name}', email='{student_email}', department='{department}'")
    
    
    try:
        # Initialize Supabase client
        supabase = get_supabase_client()
        print("DEBUG: Supabase client initialized")
        
        logger.info(f"Submitting complaint for student: {student_name}")
        
        # Prepare the data in the same format as the frontend
        complaint_data = {
            "student_name": student_name,
            "student_email": student_email,
            "department": department,
            "complaint_text": complaint_text,
            "status": "Pending"
        }
        
        # Insert the complaint using the same approach as the frontend
        response = supabase.table("student_complaints").insert([complaint_data]).execute()
        print(f"DEBUG: Supabase insert completed. Response type: {type(response)}")
        print(f"DEBUG: Response data: {response.data}")
        
        # Check for errors in the response
        if hasattr(response, 'error') and response.error:
            raise Exception(response.error.message)
        
        if not response.data:
            logger.error("Failed to submit complaint - no data returned")
            return "❌ Failed to submit your complaint. No data was returned from the server."
        
        # Return success message
        complaint_id = response.data[0]['id']
        success_message = f"✅ Your complaint has been submitted successfully. Your complaint ID is {complaint_id}. We will review it and get back to you soon."
        logger.info(f"Complaint submitted successfully with ID: {complaint_id}")
        return success_message
        
    except Exception as e:
        print(f"DEBUG: Exception in submit_complaint: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error submitting complaint: {str(e)}", exc_info=True)
        
        # Check if the error is related to RLS policies
        if "row-level security policy" in str(e):
            return "❌ There's a permission issue with the complaint system. The administrator needs to update the security policies for the complaint system. Please try again later or contact support."
        
        return f"❌ Error submitting complaint: {str(e)}"


@function_tool()
async def manual_complaint(context: RunContext) -> str:
    """
    Handles manual complaint submission by collecting user information and complaint details.
    This function guides the user through the complaint submission process step by step.
    """



    try:
        global Room
        message_data = {
            "headline": "Submit a Complaint",
            "text": "Please fill out the form below to submit your complaint. Your feedback is important to us.",
            "link": "/StudentAssistance/sumbitpage",
            "linkText": "Go to Complaint Form"
        }
        import json
        message_text = json.dumps(message_data)

        # ✅ Correct access pattern:
        # In AgentSession, you must connect manually if you want the rtc.Room instance.
        

        # ✅ Send message to frontend
        info = await Room.local_participant.send_text(
            message_text,
            topic="cloud-message"
        )
        logger.info(f"Sent text with stream ID: {info.stream_id}")

        # ✅ Also speak response
        await context.session.say(
            "I've opened the complaint form for you. Please fill in your details and submit your complaint."
        )

        return "Complaint form opened successfully."

    except Exception as e:
        logger.error(f"Error in manual_complaint: {str(e)}", extra={"room": context.session.room_name})
        await context.session.say(
            "Please visit the complaint section in the Student Assistance menu to submit your complaint."
        )
        return f"Sorry, I encountered an error: {str(e)}"



@function_tool()
async def get_club_details(context: RunContext, query: str) -> str:
    """
    Retrieves club details and programs from Supabase and asks the LLM to answer the user's specific query.
    The tool will determine if the user wants general club information or upcoming club programs.
    
    Args:
        context (RunContext): The LiveKit context object
        query (str): The user's question about clubs (e.g., "Tell me about clubs", "What programs are coming up?")
    
    Returns:
        str: LLM-generated response based on the clubs data
    """
    print(f"DEBUG: get_club_details called with query='{query}'")
    
    try:
        global Room
        message_data = {
            "headline": "check club details Manually",
            "text": "go to the club details section to check club information.",
            "link": "/EventsActivities/get_club_details",
            "linkText": "Go to Club section"
        }
        import json
        message_text = json.dumps(message_data)
        info = await Room.local_participant.send_text(
            message_text,
            topic="cloud-message"
        )
        logger.info(f"Sent text with stream ID: {info.stream_id}")
        # Initialize Supabase client
        supabase = get_supabase_client()
        print("DEBUG: Supabase client initialized")
        
        logger.info(f"Fetching club details for query: {query}")
        
        # Fetch all clubs
        clubs_response = supabase.table("clubs").select("*").execute()
        print(f"DEBUG: Clubs query completed. Response type: {type(clubs_response)}")
        print(f"DEBUG: Clubs response data: {clubs_response.data}")
        
        if not clubs_response.data:
            logger.warning("No clubs found.")
            return "There are no clubs registered at Shree Devi College."
        
        clubs_data = clubs_response.data
        logger.info(f"Retrieved {len(clubs_data)} clubs")
        
        # Fetch all upcoming club programs
        from datetime import date
        today = date.today().isoformat()
        programs_response = (
            supabase.table("club_programs")
            .select("*")
            .gte("program_date", today)  # Only get upcoming programs
            .order("program_date", desc=False)
            .execute()
        )
        print(f"DEBUG: Club programs query completed. Response type: {type(programs_response)}")
        print(f"DEBUG: Club programs response data: {programs_response.data}")
        
        programs_data = programs_response.data if programs_response.data else []
        logger.info(f"Retrieved {len(programs_data)} upcoming club programs")
        
        # Convert data to JSON
        import json
        clubs_json = json.dumps(clubs_data, indent=2)
        programs_json = json.dumps(programs_data, indent=2)
        
        # Craft LLM instruction
        llm_instruction = f"""
        The user asked: "{query}"
        
        Here is the list of all clubs at Shree Devi College:
        {clubs_json}
        
        Here is the list of upcoming club programs (only programs on or after today):
        {programs_json}
        
        Please respond in a friendly, clear, and concise way.
        - If the user is asking about general club information (e.g., "Tell me about clubs", "What clubs are available?"), 
          provide a summary of all clubs, their descriptions, heads, and contact information.
        - If the user is asking about upcoming programs or events (e.g., "What club events are coming up?", "Any programs this month?"), 
          focus on the upcoming programs, including dates, times, venues, and which clubs are organizing them.
        - If the user is asking about a specific club, provide details about that club and any upcoming programs it's organizing.
        - If the user hasn't specified whether they want general club info or upcoming programs, ask them: 
          "Would you like to know about our clubs in general, or are you interested in upcoming club programs?"
        - Do NOT invent details. Use only the provided data.
        """
        
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        logger.debug(f"LLM generated answer for club details query: {query}")
        return llm_instruction
        
    except Exception as e:
        print(f"DEBUG: Exception in get_club_details: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error retrieving club details: {str(e)}", exc_info=True)
        return f"❌ Error retrieving club details: {str(e)}"


@function_tool()
async def get_upcoming_events(context: RunContext, query: str) -> str:
    """
    Retrieves upcoming college events from the Supabase 'events' table and asks the LLM to answer the user's specific query.
    The tool supports filtering by department if specified in the query, but first prompts the user to clarify if needed.

    Args:
        context (RunContext): The LiveKit context object
        query (str): The user's question about upcoming events (e.g., "What events are coming up?", "Any CS department events?")

    Returns:
        str: LLM-generated response based on the events data
    """
    print(f"DEBUG: get_upcoming_events called with query='{query}'")

    try:
        global Room
        message_data = {
            "headline": "check Event details Manually",
            "text": "go to the Event section to check upcoming event.",
            "link": "/EventsActivities/UpcomingEvents",
            "linkText": "Go to Event section"
        }
        import json
        message_text = json.dumps(message_data)
        info = await Room.local_participant.send_text(
            message_text,
            topic="cloud-message"
        )
        logger.info(f"Sent text with stream ID: {info.stream_id}")
        # Initialize Supabase client
        supabase = get_supabase_client()
        print("DEBUG: Supabase client initialized")

        logger.info(f"Fetching upcoming events for query: {query}")

        # Fetch all upcoming events (event_date >= today)
        from datetime import date
        today = date.today().isoformat()
        print(f"DEBUG: Filtering events on or after {today}")

        response = (
            supabase.table("events")
            .select("*")
            .gte("event_date", today)
            .order("event_date", desc=False)
            .execute()
        )

        print(f"DEBUG: Supabase query completed. Response type: {type(response)}")
        print(f"DEBUG: Response data: {response.data}")

        if not response.data:
            logger.warning("No upcoming events found.")
            return "There are no upcoming events scheduled at Shree Devi College."

        events_data = response.data
        logger.info(f"Retrieved {len(events_data)} upcoming events")

        import json
        events_json = json.dumps(events_data, indent=2)

        # Craft LLM instruction
        llm_instruction = f"""
        The user asked: "{query}"

        Here is the list of upcoming college events (only events on or after today):
        {events_json}

        Please respond in a friendly, clear, and concise way.
        - If the user asked about a specific department (e.g., 'CSE', 'Mechanical', 'MBA'), only mention events where the 'department' field matches or is null (college-wide).
        - If the user didn't specify a department, summarize all events briefly.
        - If no events match the user's department, say so clearly.
        - Always include event name, date, time, venue, and registration link (if available).
        - Do NOT invent details. Use only the provided data.
        """

        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        logger.debug(f"LLM generated answer for upcoming events query: {query}")
        return llm_instruction

    except Exception as e:
        print(f"DEBUG: Exception in get_upcoming_events: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error retrieving upcoming events: {str(e)}", exc_info=True)
        return f"❌ Error retrieving upcoming events: {str(e)}"


@function_tool()
async def get_exam_timetable(
    context: RunContext,
    query: str
) -> str:
    """
    Retrieves exam timetable data and asks the LLM to answer the user's specific query based on it.
    
    Args:
        context (RunContext): The LiveKit context object
        query (str): The specific question or topic the user asked about the exam timetable
    
    Returns:
        str: LLM-generated response based on the exam timetable data
    """
    print(f"DEBUG: get_exam_timetable called with query='{query}'")
    
    try:
        global Room
        message_data = {
            "headline": "check your Exam details Manually",
            "text": "go to the Exam timetable details section to check upcoming exams.",
            "link": "/AdmissionsAcademics/examtimetable",
            "linkText": "Go to Exam section"
        }
        import json
        message_text = json.dumps(message_data)
        info = await Room.local_participant.send_text(
            message_text,
            topic="cloud-message"
        )
        logger.info(f"Sent text with stream ID: {info.stream_id}")
        # Initialize Supabase client
        supabase = get_supabase_client()
        print("DEBUG: Supabase client initialized")
        
        # Log the attempt
        logger.info(f"Fetching exam timetable for query: {query}")
        
        # Build and execute the query
        print("DEBUG: Executing Supabase query...")
        response = supabase.table("exam_timetable").select("*").execute()
        print(f"DEBUG: Supabase query completed. Response type: {type(response)}")
        print(f"DEBUG: Response data: {response.data}")
        
        # Check if we got data
        if not response.data:
            print("DEBUG: No data found in response")
            logger.warning(f"No exam timetable data found for query: {query}")
            return f"No exam timetable data found for query: {query}."
        
        # Format the results as JSON for LLM processing
        timetable_data = response.data
        print(f"DEBUG: Found {len(timetable_data)} records")
        logger.info(f"Retrieved {len(timetable_data)} exam entries")
        
        # Convert data to JSON string
        import json
        timetable_json = json.dumps(timetable_data, indent=2)
        
        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the complete exam timetable data in JSON format:
        {timetable_json}
        
        Please answer the user's question based *strictly* and in human sentence on the provided data. 
        If the information is not available in the data, state that clearly.
        Format your response in a clear, readable way with appropriate sections.
        """

        # Use the session's generate_reply to ask the LLM to process the data
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug(f"LLM generated answer for query: {query}")
        return llm_instruction
        
    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error retrieving exam timetable: {str(e)}", exc_info=True)
        return f"❌ Error retrieving exam timetable: {str(e)}"


@function_tool()
async def get_syllabus_summary(context: RunContext, query: str):
    """
    Retrieves the full college syllabus summary data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college syllabus summary.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_syllabus_summary.json")
    logger.info("get_syllabus_summary called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college syllabus summary data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college syllabus summary data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college syllabus summary JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college syllabus summary 
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college syllabus summary JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def get_scholarship_info(context: RunContext, query: str):
    """
    Retrieves the full college scholarship information data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college scholarship information.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_scholarship_info.json")
    logger.info("get_scholarship_info called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college scholarship information data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college scholarship information data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college scholarship information JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college scholarship information 
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college scholarship information JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string

@function_tool()
async def get_recruiters_list(context: RunContext, query: str):
    """
    Retrieves the full college recruiters list data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college recruiters list.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_recruiters_list.json")
    logger.info("get_recruiters_list called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college recruiters list data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college recruiters list data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college recruiters list JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college recruiters list 
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college recruiters list JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string

@function_tool()
async def get_placement_stats(context: RunContext, query: str):
    """
    Retrieves the full college placement statistics data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college placement statistics.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_placement_stats.json")
    logger.info("get_placement_stats called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college placement statistics data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college placement statistics data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college placement statistics JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college placement statistics 
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college placement statistics JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def get_library_timing(context: RunContext, query: str):
    """
    Retrieves the full college library timing data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college library timing.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_library_timing.json")
    logger.info("get_library_timing called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college library timing data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college library timing data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college library timing JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college library timing 
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college library timing JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string



@function_tool()
async def get_hostel_rules(context: RunContext, query: str):
    """
    Retrieves the full college hostel rules data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college hostel rules.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_hostel_rules.json")
    logger.info("get_hostel_rules called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college hostel rules data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college hostel rules data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college hostel rules JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college hostel rules 
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college hostel rules JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def get_faculty_details(context: RunContext, query: str):
    """
    Retrieves the full college faculty details data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college faculty details.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_faculty_details.json")
    logger.info("get_faculty_details called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college faculty details data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college faculty details data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college faculty details JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college faculty details 
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college faculty details JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def get_facilities_info(context: RunContext, query: str):
    """
    Retrieves the full college facilities information data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college facilities information.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_facilities_info.json")
    logger.info("get_facilities_info called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college facilities information data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college facilities information data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college facilities information JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college facilities information data:
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college facilities information JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def get_departments_list(context: RunContext, query: str):
    """
    Retrieves the full college departments list data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college departments list.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_departments_list.json")
    logger.info("get_departments_list called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college departments list data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college departments list data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college departments list JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college departments list data:
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college departments list JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def get_course_details(context: RunContext, query: str):
    """
    Retrieves the full college course details data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college course details.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_course_details.json")
    logger.info("get_course_details called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college course details data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college course details data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college course details JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college course details data:
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        logger.debug("LLM generated answer for  is '%s'.", answer_from_llm)
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college course details JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def get_cafeteria_menu(context: RunContext, query: str):
    """
    Retrieves the full college cafeteria menu information data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college cafeteria menu.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_cafeteria_menu.json")
    logger.info("get_cafeteria_menu called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college cafeteria menu data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college cafeteria menu data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college cafeteria menu JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college cafeteria menu data:
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college cafeteria menu JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def get_accreditation_info(context: RunContext, query: str):
    """
    Retrieves the full college accreditation information data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college accreditation information.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_accreditation_info.json")
    logger.info("get_accreditation_info called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college accreditation information data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college accreditation information data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college accreditation information JSON for query '%s', bytes=%d", query, len(content))

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college accreditation information data:
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college accreditation information JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def get_college_overview(context: RunContext, query: str):
    """
    Retrieves the full college overview data and asks the LLM to answer the user's specific query based on it.
    Args:
        context (RunContext): The context of the function call, including the session.
        query (str): The specific question or topic the user asked about the college overview.
    """
    json_path = os.path.join(os.path.dirname(__file__), "data", "dat1", "get_college_overview.json")
    logger.info("get_college_overview called for query: '%s' (room=%s) path=%s", 
                query, getattr(context, "session", None) and getattr(context.session, "room", None) or context, json_path)

    if not os.path.exists(json_path):
        msg = f"I attempted to load the college overview data but the file was not found: {json_path}"
        logger.error(msg)
        return msg # Return the error message string

    try:
        def _read():
            with open(json_path, "r", encoding="utf-8") as f:
                return f.read()

        content = await asyncio.to_thread(_read)

        if not content or not content.strip():
            msg = "I attempted to load the college overview data but it was empty."
            logger.error(msg)
            return msg # Return the error message string

        # Log size and a short preview for debugging
        logger.info("Loaded college overview JSON for query '%s', bytes=%d", query, len(content))
        # Optionally, truncate content for logging if it's very large
        # preview = content if len(content) <= 2000 else content[:2000] + "\n...TRUNCATED..."
        # logger.debug("college overview preview:\n%s", preview)

        # Prepare the instruction for the LLM
        llm_instruction = f"""
        The user asked: "{query}"
        Here is the full college overview data:
        {content}
        Please answer the user's question based *strictly* on the provided data. If the information is not available in the data, state that clearly.
        """

        # Use the session's generate_reply to ask the LLM to process the data and answer the query
        # This call sends the instruction to the LLM associated with the session
        # It returns the LLM's response (the answer based on the data)
        answer_from_llm = await context.session.generate_reply(instructions=llm_instruction)
        
        logger.debug("LLM generated answer for query '%s'.", query)
        # Return the answer generated by the LLM based on the data
        return llm_instruction # Return the string answer

    except Exception as e:
        logger.exception("Failed to load or process college overview JSON for query '%s': %s", query, e)
        error_msg = f"I encountered an error while trying to find information about '{query}': {e}"
        return error_msg # Return the error message string


@function_tool()
async def my_tool(context: RunContext):
    """A simple tool that responds with 'Good boy'. Use this when the user specifically asks to use the tool."""
    logger.debug("my_tool called")
    await context.session.say("Good boy")

class CollegeAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful, friendly, and precise AI assistant for Shree Devi College.\n"
                "IMPORTANT RULES (follow exactly):\n"
                "1) If the user asks about the college history, vision, mission, campuses, or any overview topic, "
                "YOU MUST CALL the tool get_college_overview() with the user's specific question and wait for its result before replying.\n"
                "2) If the user asks about accreditation, NAAC grade, NBA status, AICTE approval, ISO certification, or any accreditation-related topic, "
                "YOU MUST CALL the tool get_accreditation_info() with the user's specific question and wait for its result before replying.\n"
                "3) If the user asks about cafeteria, canteen, food menu, or any food-related facilities, "
                "YOU MUST CALL the tool get_cafeteria_menu() with the user's specific question and wait for its result before replying.\n"
                "4) If the user asks about courses, programs, departments, B.E., M.Tech, MBA, B.Sc, M.Sc, or any academic programs, "
                "YOU MUST CALL the tool get_course_details() with the user's specific question and wait for its result before replying.\n"
                "5) If the user asks about specific departments, their names, contact details, campus locations, or department-specific information, "
                "YOU MUST CALL the tool get_departments_list() with the user's specific question and wait for its result before replying.\n"
                "6) If the user asks about campus facilities, hostels, library, transport, sports, health center, lecture halls, Wi-Fi, or any infrastructure facilities, "
                "YOU MUST CALL the tool get_facilities_info() with the user's specific question and wait for its result before replying.\n"
                "7) If the user asks about faculty members, professors, department heads, college leadership, or any personnel information, "
                "YOU MUST CALL the tool get_faculty_details() with the user's specific question and wait for its result before replying.\n"
                "8) If the user asks about hostel rules, policies, facilities, amenities, security, or accommodation details, "
                "YOU MUST CALL the tool get_hostel_rules() with the user's specific question and wait for its result before replying.\n"
                "9) If the user asks about library timing, hours of operation, or library-specific information, "
                "YOU MUST CALL the tool get_library_timing() with the user's specific question and wait for its result before replying.\n"
                "10) If the user asks about placement statistics, placement rates, packages, companies, or any placement-related information, "
                "YOU MUST CALL the tool get_placement_stats() with the user's specific question and wait for its result before replying.\n"
                "11) If the user asks about company recruiters, list of companies visiting campus, or recruiter-specific information, "
                "YOU MUST CALL the tool get_recruiters_list() with the user's specific question and wait for its result before replying.\n"
                "12) If the user asks about scholarships, fee structure, government scholarships, or financial aid information, "
                "YOU MUST CALL the tool get_scholarship_info() with the user's specific question and wait for its result before replying.\n"
                "13) If the user asks about syllabus, curriculum, course structure, or academic content details, "
                "YOU MUST CALL the tool get_syllabus_summary() with the user's specific question and wait for its result before replying.\n"
                "14) If the user asks about exam dates, exam schedule, exam timetable, when exams are held, or subject-wise exam timing, "
                "YOU MUST FIRST ask the user which department they need the exam timetable for, and which semester. "
                "Once the user provides both the department and semester, THEN CALL the tool get_exam_timetable() with these parameters and wait for its result before replying.\n"
                "15) If the user asks about upcoming college events, fests, workshops, seminars, guest lectures, cultural/technical activities, event schedules, or registration links for any event, "
                "YOU MUST FIRST ask the user: \"Do you want to see events for a specific department, or all upcoming events?\" "
                "Once the user clarifies (e.g., \"CSE events\", \"MBA workshop\", or \"all events\"), THEN CALL the tool get_upcoming_events() with the user's exact clarified question and wait for its result before replying.\n"
                "16) If the user asks about clubs, student organizations, club activities, or any club-related information, "
                "YOU MUST CALL the tool get_club_details() with the user's specific question and wait for its result before replying.\n"
                "17) If the user wants to submit a complaint, YOU MUST FIRST ask the user: \"Do you want to submit a complaint using manual text input or would you like to say the details and I'll raise the complaint for you?\" "
                "If the user chooses manual text input, YOU MUST CALL the tool manual_complaint(). "
                "If the user wants to say the details, YOU MUST collect the required information step-by-step: "
                "First, ask for the user's name. Once you get it, ask for their email address. "
                "After receiving the email, ask for their department. "
                "Finally, ask for their complaint details. "
                "Once you have collected all four pieces of information (name, email, department, and complaint details), "
                "THEN CALL the tool submit_complaint() with these parameters and wait for its result before replying.\n"
                "18) If the user wants to contact the helpdesk or request a callback, YOU MUST FIRST ask the user: \"Do you want to request a callback using manual text input or would you like to say the details and I'll contact the helpdesk for you?\" "
                "If the user chooses manual text input, YOU MUST CALL the tool manual_callback(). "
                "If the user wants to say the details, YOU MUST collect the required information step-by-step: "
                "First, ask for the user's name. Once you get it, ask for their email address. "
                "After receiving the email, ask for their phone number. "
                "Finally, ask for their request details. "
                "Once you have collected all four pieces of information (name, email, phone number, and request details), "
                "THEN CALL the tool contact_helpdesk() with these parameters and wait for its result before replying.\n"
                "19) Use the result from the appropriate tool as your answer. The tool will return a summary or specific answer based on the college data.\n"
                "20) If a tool returns an error message, repeat it to the user.\n"
                "21) Only discuss Shree Devi College. Redirect unrelated questions politely.\n"
                "22) Be factual, concise, and avoid repeating yourself."
            ),
            tools=[
                my_tool,
                get_college_overview,
                get_accreditation_info,
                get_cafeteria_menu,
                get_course_details,
                get_departments_list,
                get_facilities_info,
                get_faculty_details,
                get_hostel_rules,
                get_library_timing,
                get_placement_stats,
                get_recruiters_list,
                get_scholarship_info,
                get_syllabus_summary,
                get_exam_timetable,
                get_upcoming_events,
                get_club_details,
                submit_complaint,
                manual_complaint,
                contact_helpdesk,  
                manual_callback,  
            ],
        )

    

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    global Room
    Room =ctx.room
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Starting session for room: {ctx.room.name}")

    session = AgentSession(
        stt=assemblyai.STT(),
        llm=google.LLM(
            model="gemini-2.5-flash",
            temperature=0.3,
        ),
        # tts=ElevenLabsTTS(
        #     model="eleven_multilingual_v2",
        #     voice_id="ODq5zmih8GrVes37Dizd",
        # ),
        tts=inference.TTS(
        model="cartesia/sonic-2", 
        voice="79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e", 
        language="en"
        ),
    )

    await session.start(
        agent=CollegeAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and offer specific help regarding Shree Devi College.in short sentences."
    )

    await ctx.connect()

    # background_audio = BackgroundAudioPlayer(
    #     thinking_sound=[
    #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6),
    #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.5),
    #     ],
    # )
    # await background_audio.start(room=ctx.room, agent_session=session)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
