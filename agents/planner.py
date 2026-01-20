"""
Planner Agent (Agent 1)
Analyzes user questions and data schema to create execution plans.
The "Thinking" agent that determines what needs to be done.
"""

import os
from google import genai
from google.genai import types
from typing import Dict, Any
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class PlannerAgent:
    """
    Agent 1: The Planner
    Analyzes user questions and creates step-by-step execution plans.
    """
    
    SYSTEM_PROMPT = """You are the Planner Agent in a multi-agent data analysis system.
Your role is to analyze user questions about their data and create clear, actionable execution plans.

## Your Responsibilities:
1. Understand the user's question in the context of their data schema
2. Consider previous conversation for follow-up questions
3. Create a step-by-step plan for data analysis
4. Determine if visualization is needed and what type

## Output Format:
You MUST respond with a structured plan in this exact format:

**Understanding:** [Brief restatement of what the user wants]

**Plan:**
1. [First step - e.g., "Filter data where..."]
2. [Second step - e.g., "Group by Category and sum Sales"]
3. [Third step - e.g., "Sort by value descending"]
4. [Final step - e.g., "Return top 5 results"]

**Visualization:** [NONE | bar | line | pie | scatter | area | count | map]
If a chart is needed, specify the chart type. Use NONE if no visualization needed.

**Chart Config:**
- x_column: [column name for x-axis, or location column for maps]
- y_column: [column name for y-axis, or value column for maps]
- title: [descriptive chart title]
- color_by: [optional grouping column, or NONE]

## Guidelines:
- For trends over time, recommend LINE charts (use date/time column for x_column)
- For comparisons between categories or rankings (top N), recommend BAR charts
- For distributions/proportions, recommend PIE charts
- For correlations, recommend SCATTER charts
- For count-based questions, recommend COUNT plots
- For geographic/location-based data (cities, states, countries), recommend MAP charts

## Important:
- Always reference actual column names from the schema
- For time-based analysis, use 'Order Date' or similar date column as x_column
- For map charts, use location column (City, State, Country) as x_column
- Consider data types when planning operations
- Be specific about aggregation functions (sum, mean, count, etc.)
- For follow-up questions, use context from previous conversation
"""

    def __init__(self):
        """Initialize the Planner agent with Gemini API."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize the new google.genai client
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-flash-lite-latest"
    
    def create_plan(
        self,
        question: str,
        schema_info: str,
        conversation_context: str = ""
    ) -> Dict[str, Any]:
        """
        Create an execution plan for the user's question.
        
        Args:
            question: User's natural language question
            schema_info: Formatted schema information of the dataset
            conversation_context: Previous conversation for context
            
        Returns:
            Dictionary with plan details
        """
        # Build the prompt
        prompt = f"""## Current Data Schema:
{schema_info}

## Conversation Context:
{conversation_context if conversation_context else "This is the first question."}

## User Question:
{question}

Create your execution plan now:"""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=1024
                )
            )
            plan_text = response.text
            
            # Parse the plan
            parsed = self._parse_plan(plan_text)
            parsed['raw_plan'] = plan_text
            parsed['question'] = question
            
            return parsed
            
        except Exception as e:
            return {
                'error': str(e),
                'raw_plan': f"Error creating plan: {e}",
                'understanding': "Could not process the question",
                'steps': [],
                'visualization': 'NONE',
                'chart_config': {}
            }
    
    def _parse_plan(self, plan_text: str) -> Dict[str, Any]:
        """
        Parse the structured plan from the LLM response.
        
        Args:
            plan_text: Raw text from LLM
            
        Returns:
            Parsed plan dictionary
        """
        result = {
            'understanding': '',
            'steps': [],
            'visualization': 'NONE',
            'chart_config': {}
        }
        
        lines = plan_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('**Understanding:**'):
                result['understanding'] = line.replace('**Understanding:**', '').strip()
                current_section = 'understanding'
            elif line.startswith('**Plan:**'):
                current_section = 'plan'
            elif line.startswith('**Visualization:**'):
                viz = line.replace('**Visualization:**', '').strip().lower()
                if viz and viz != 'none':
                    result['visualization'] = viz
                current_section = 'visualization'
            elif line.startswith('**Chart Config:**'):
                current_section = 'chart_config'
            elif current_section == 'plan' and line and line[0].isdigit():
                # Extract step text (remove number and period)
                step_text = line.split('.', 1)[-1].strip()
                if step_text:
                    result['steps'].append(step_text)
            elif current_section == 'chart_config' and line.startswith('-'):
                # Parse chart config options
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    if value.lower() != 'none':
                        result['chart_config'][key] = value
        
        return result
    
    def get_reasoning(self, plan: Dict[str, Any]) -> str:
        """
        Get a human-readable explanation of the plan.
        
        Args:
            plan: Parsed plan dictionary
            
        Returns:
            Formatted reasoning string
        """
        parts = ["📋 **Execution Plan**\n"]
        
        if plan.get('understanding'):
            parts.append(f"**Understanding:** {plan['understanding']}\n")
        
        if plan.get('steps'):
            parts.append("**Steps:**")
            for i, step in enumerate(plan['steps'], 1):
                parts.append(f"  {i}. {step}")
        
        if plan.get('visualization') and plan['visualization'] != 'NONE':
            parts.append(f"\n**Visualization:** {plan['visualization'].upper()} chart")
            
            if plan.get('chart_config'):
                config = plan['chart_config']
                if config.get('title'):
                    parts.append(f"  Title: {config['title']}")
        
        return '\n'.join(parts)
