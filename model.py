from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.tool_context import ToolContext
from google.genai import types


retry_config=types.HttpRetryOptions(
    attempts=5,  
    exp_base=7, 
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)


def save_userinfo(
    tool_context: ToolContext, location: str
) -> Dict[str, Any]:
    """
    Tool to record and save user location in session state.

    Args:
        Location: Free-form text field. Can be a city name, district, or full address with street.
    """
    tool_context.state["user:location"] = location

    return {"status": "success"}

def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user location from session state.
    """
    location = tool_context.state.get("user:location", "Location not found")

    return {"status": "success", "location": location}



APP_NAME = "agents"
USER_ID = "default"
SESSION_ID = "session1"

location_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_config=retry_config),
    name="location_agent",
    instruction="""Ти асистент який зберігає локацію. Твоя задача лише оперувати даними про локацію (зберігати та діставати) ЛИШЕ за допомогою наданих двох інструментів.
    1. Якщо користувач явно вказав локацію, то збережи її за допомогою save_userinfo(location: str).
    2. Якщо потрібно отримати локацію - отримуємо за допомогою retrieve_userinfo().""",
    tools=[FunctionTool(save_userinfo), FunctionTool(retrieve_userinfo)], 
)

search_agent = Agent(
    name='search_agent',
    model=Gemini(model="gemini-2.5-flash-lite", retry_config=retry_config),
    instruction=(
    """Ти асистент, який шукає погоду за допомогою google_search, а саме бери інформацію надану Google Weather.
    Якщо не вимагається якась конкретна інформація про погоду, то надавай загальну інформацію (денну температуру) та важливі погодні умови (ймовірність опадів, гололід, сильний вітер тощо).
    Якщо не можеш знайти інформацію або не впевнений в ній, не надавай її.
    """
    ),
    tools=[google_search],
)

root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_config=retry_config),
    name="text_chat_bot",
    description="A text chatbot.",
    instruction="""Ти друг-асистент, який може підказати погоду.
    Використовуй location_agent аби оперувати збереженими даними про локацію:
    1. Має бути збережена остання згадка про локацію. Тобто якщо збережена одна локація і тепер користувача цікавить інша - потрібно зберегти нову.
    2. Якщо немає збережених локацій та користувач її не надав, попроси уточнити, де саме його цікавить погода. 

    Використовуй search_agent для пошуку погоди:
    1. Відповідь має бути короткою. Якщо користувач не просить конкретної інформації, то надай лише температуру ВДЕНЬ та важливі погодні умови (якщо такі є): опади, сильний вітер, гололід. 
    2. Якщо погодні умови в межах норми (відсутність опадів, звичайний вітер та відсутність гололіду), то достатньо надати лише температуру. Хмарно чи ясно - не потрібно казати, якщо користувач цього не просить.
    3. Якщо користувач надає некоректний запит, або питання не про погоду, то відповідь має бути "На жаль, я можу допомогти лише з питаннями про погоду" або прохання уточнення.
    4. Якщо користувача цікавить щось конкретне про погоду - надай цю конкретну інформацію. Проте відповідь має бути чіткою.
    Приклади відповідей:  
    \nЗавтра вдень температура буде близько 3°C. Можливо буде сильний дощ приблизно з 14:00 до 16:00. 
    \nСьогодні температура від 14°C до 20°C протягом дня. Ввечері буде сильний вітер (20 м/с).
    Відповідь має бути 1-2 речення максимум. Дуже коротко та чітко!
    """,
    tools=[AgentTool(agent=search_agent), AgentTool(agent=location_agent)], 
)


