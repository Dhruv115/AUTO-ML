from crewai import Agent
from langchain_groq import ChatGroq
from tools import load_csv, run_eda, preprocess_data, select_and_train_models, tune_best_model, generate_report

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )

def create_agents():
    llm = get_llm()

    eda_agent = Agent(
        role="Data Analyst",
        goal="Load the CSV file and run EDA ONCE. Call load_csv tool ONCE, then run_eda tool ONCE, then immediately provide your Final Answer. Do not repeat any tool calls.",
        backstory="Expert data analyst who uncovers hidden patterns and data quality issues before any modeling. Always provides Final Answer immediately after getting tool results.",
        tools=[load_csv, run_eda],
        llm=llm,
        verbose=True,
        max_iter=4,
        max_rpm=10,
    )

    preprocessor_agent = Agent(
        role="Data Preprocessor",
        goal="Call preprocess_data tool ONCE with strategy='median'. After receiving the result, immediately provide your Final Answer. Do not call the tool more than once under any circumstances.",
        backstory="Specialist in data wrangling who handles missing values, encoding, and feature scaling. Always provides Final Answer immediately after the first tool result.",
        tools=[preprocess_data],
        llm=llm,
        verbose=True,
        max_iter=3,
        max_rpm=10,
    )

    model_agent = Agent(
        role="ML Engineer",
        goal="Call select_and_train_models tool ONCE. After receiving accuracy scores, immediately provide your Final Answer listing all model scores and the best model. Do not call the tool more than once.",
        backstory="Experienced ML engineer who benchmarks models systematically. Always provides Final Answer immediately after getting tool results.",
        tools=[select_and_train_models],
        llm=llm,
        verbose=True,
        max_iter=3,
        max_rpm=10,
    )

    tuning_agent = Agent(
        role="Hyperparameter Tuning Specialist",
        goal="Call tune_best_model tool ONCE with dummy='run'. After receiving tuning results, immediately provide your Final Answer with best params and accuracy. Do not call the tool more than once.",
        backstory="Optimization expert who squeezes every bit of performance out of ML models. Always provides Final Answer immediately after the first tool result.",
        tools=[tune_best_model],
        llm=llm,
        verbose=True,
        max_iter=3,
        max_rpm=10,
    )

    report_agent = Agent(
        role="ML Report Writer",
        goal="Call generate_report tool ONCE with output_path='automl_report.md'. After receiving the report, immediately provide your Final Answer with the report content. Do not call the tool more than once.",
        backstory="Technical writer with ML expertise who communicates results clearly. Always provides Final Answer immediately after the first tool result.",
        tools=[generate_report],
        llm=llm,
        verbose=True,
        max_iter=3,
        max_rpm=10,
    )

    return eda_agent, preprocessor_agent, model_agent, tuning_agent, report_agent