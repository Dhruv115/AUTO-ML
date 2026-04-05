from crewai import Task

def create_tasks(agents, csv_path: str, target_column: str):
    eda_agent, preprocessor_agent, model_agent, tuning_agent, report_agent = agents

    eda_task = Task(
        description=f"""
        Step 1: Call load_csv tool ONCE with filepath="{csv_path}"
        Step 2: Call run_eda tool ONCE with target_column="{target_column}"
        Step 3: Immediately write your Final Answer with the EDA results.
        DO NOT repeat any tool calls. Two tool calls total, then Final Answer.
        """,
        expected_output="A detailed EDA summary including data shape, null counts, target distribution, and key statistics.",
        agent=eda_agent,
    )

    preprocess_task = Task(
        description="""
        Step 1: Call preprocess_data tool ONCE with strategy="median"
        Step 2: Immediately write your Final Answer with the preprocessing results.
        DO NOT call the tool again. One tool call total, then Final Answer.
        """,
        expected_output="Confirmation of preprocessing steps completed and final feature matrix shape.",
        agent=preprocessor_agent,
        context=[eda_task],
    )

    model_task = Task(
        description="""
        Step 1: Call select_and_train_models tool ONCE.
        Step 2: Immediately write your Final Answer listing all model accuracies and the best model.
        DO NOT call the tool again. One tool call total, then Final Answer.
        """,
        expected_output="Accuracy scores for all models, clearly identifying the best performing model.",
        agent=model_agent,
        context=[preprocess_task],
    )

    tuning_task = Task(
        description="""
        Step 1: Call tune_best_model tool ONCE with dummy="run"
        Step 2: Immediately write your Final Answer with best params, base vs tuned accuracy, and classification report.
        DO NOT call the tool again. One tool call total, then Final Answer.
        """,
        expected_output="Best hyperparameters, comparison of base vs tuned accuracy, and classification report.",
        agent=tuning_agent,
        context=[model_task],
    )

    report_task = Task(
        description="""
        Step 1: Call generate_report tool ONCE with output_path="automl_report.md"
        Step 2: Immediately write your Final Answer with the complete report content.
        DO NOT call the tool again. One tool call total, then Final Answer.
        """,
        expected_output="A saved markdown report file summarizing the complete AutoML pipeline results.",
        agent=report_agent,
        context=[tuning_task],
    )

    return [eda_task, preprocess_task, model_task, tuning_task, report_task]