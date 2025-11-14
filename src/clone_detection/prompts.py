import textwrap

def get_system_prompt_for_direct_clone_detection():
    return textwrap.dedent("""
        You are an expert cross-language code clone detection specialist.
        Your task is to determine if two code snippets of different programming languages are clones by analyzing their functionality, logic, and semantic equivalence.

        Focus on semantic similarity and functional equivalence rather than syntactic differences.
    """).strip()

def get_prompt_for_direct_clone_detection(code1, code_lang1, code2, code_lang2):
    prompt = textwrap.dedent(f"""
        {code_lang1} code:
        {code1}
        
        {code_lang2} code:
        {code2}
        
        Let's think step-by-step and analyze if the {code_lang1} and {code_lang2} code snippets are semantically equivalent by following these steps:
        Step 1: Structure Analysis: Analyze the structure of both codes and identify main components, control flow, data types, and dependencies in each        
        Step 2: Functional Equivalence: Evaluate if both solve the same problem with equivalent functionality        
        Step 3: Semantic Comparison: Ignore syntax differences, variable names, and language-specific constructs and evaluate the semantic equivalence
        
        Provide your response in this exact format:
        RATIONALE:
        [Clear explanation of why you conclude they are clones or not clones, referencing specific code elements]
        
        DECISION:
        [Either CLONE or NOT_CLONE - nothing else]
    """).strip()

    return prompt

def get_system_prompt_for_algorithm_based_clone_detection():
    return textwrap.dedent("""
        You are an expert cross-language code clone detection specialist working with algorithm-based analysis.
        You work in two phases:
        1. Extract detailed algorithms by capturing all logic and control flow from code snippets of different languages
        2. Compare the algorithms to determine if the original codes are functionally/semantically equivalent or not.
        
        Focus on identifying functional equivalence through algorithmic similarity rather than syntactic patterns.
    """).strip()

def get_prompt_to_generate_algorithm_from_code(code, code_lang):
    prompt = textwrap.dedent(f"""
        {code_lang} code:
        {code}

        Let's think step-by-step and extract a detailed algorithm from the above {code_lang} code using the following instructions:
        Step 1 - Code Structure: Identify function signatures, classes, and program organization.
        Step 2 - Core Logic: Extract execution flow, control structures, data transformations, and I/O operations.
        Step 3 - Algorithm Components: Document data types, dependencies, and exact conditions.
        
        Provide only the detailed algorithm in a language-agnostic format. Do not include any headers, comments, explanations, or examples.
    """).strip()

    return prompt

def get_prompt_for_algorithm_based_clone_detection(algorithm1, algorithm2):
    prompt = textwrap.dedent(f"""
        Algorithm 1:
        {algorithm1}
        
        Algorithm 2:
        {algorithm2}
        
        Let's think step-by-step and determine if the above two algorithms represent equivalent functionality using the following instructions:
        Step 1: Problem Identification: Analyze both algorithm descriptions and identify the core problem each algorithm solves.
        Step 2: Approach Comparison: Compare control flow, data structures, logic patterns, and computational steps.
        Step 3: Semantic Equivalence: Determine if the algorithms implement the same solution strategy despite potential differences in notation or abstraction level.
        
        Provide your response in this exact format:        
        RATIONALE:
        [Clear explanation of why the algorithms are equivalent or different, referencing specific algorithmic elements]
        
        DECISION:
        [Either CLONE or NOT_CLONE - nothing else]
    """).strip()

    return prompt
