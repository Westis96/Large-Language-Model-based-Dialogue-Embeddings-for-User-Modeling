# Predefined string templates for generating various types of content.

# Template for guessing what a persona might ask an AI to do.
instruction_generation_template = """
Objective: Predict a likely user prompt based on the persona below.
Persona: {persona}
Instructions:
- The prompt should be detailed and specific.
- Start the response with "User prompt:".
"""

# Template for initiating a dialog, with the persona making the first statement.
dialog_start_template = """
Objective: Generate the first message a persona would send to an AI assistant.
Persona: {persona}
Instructions:
- The message should be a natural conversation starter.
- It should subtly hint at the persona's identity without revealing everything.
- Output only the message itself.
"""

# Template for continuing an ongoing dialog, with the persona responding.
dialog_continue_template = """
Objective: Generate the persona's next response in an ongoing conversation.
Persona: {persona}
Conversation History:
{conversation_history}
Instructions:
- The response must be a natural continuation of the dialog.
- Reveal new aspects of the persona.
- Output only the message itself.
"""

def get_template_by_name(template_name):
    """
    Returns a specific prompt template based on its name.
    """
    templates = {
        "instruction": instruction_generation_template,
        "dialog_start": dialog_start_template,
        "dialog_continue": dialog_continue_template,
    }
    return templates.get(template_name) 