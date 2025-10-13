"""Constants for prompt formatting in unlearning evaluation."""

# Instruction format templates for different models
GEMMA_INST_FORMAT = "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
MIXTRAL_INST_FORMAT = " [INST] {prompt} [/INST]"

# Pre-question text for different evaluation datasets
PRE_WMDP_BIO = "The following are multiple choice questions (with answers) about biology.\n"
PRE_WMDP_CYBER = (
    "The following are multiple choice questions (with answers) about cyber security.\n"
)
PRE_QUESTION_FORMAT = (
    "The following are multiple choice questions (with answers) about {subject}.\n"
)
