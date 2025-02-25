<p align="center">
<img src=./RI-SeniorAITest-Github.png>
</p>

# Test Introduction

You have been provided with the video "AICandidateTest-FINAL.mp4," which offers a top-down view of experimental work within a fume hood. The objective of this test is to evaluate your proficiency in computer vision, technical acumen, and problem-solving capabilities. You are free to tackle the challenges using your preferred programming language and any applicable tools or techniques. Treat this test as if you were undertaking a regular day of work.

# Task

The test encompasses the following objectives:

* Identify different object types in the scene (petri dishes, bottles, hands).
* Determine interactions between objects (e.g., hand touches dish, dish is filled).
* Track the count of filled dishes as the scene unfolds.
* Identify any additional vision-based insights in the scene that you find compelling.

In order to assess your ability but not consume too much of your time, please be thorough in researching methods before attempting any implementation. Prioritise the tasks above from top to bottom in terms of implementation, ensuring only to add your own insights after completing the first bullet points if you still have time.

The task should take you a minimum of 3 hours, of which we personally recommend spending around 45 minutes planning and preparing a report of your planned approaches, and the remaining time on showing off your coding prowess.

Primarily we work in Python or C++ for our AI services, so demonstrations of these skills would be ideal. If you have skills to show from other languages (alternative approaches) however, please include them in your report as we are interested in seeing your full range of skills.

# Setup & Environment

To access the video, kindly follow this link: [Video](https://reach-industries-candidate-tests.s3.eu-west-2.amazonaws.com/AICandidateTest-FINAL.mp4). In certain browsers, you might need to right-click on the link and choose "Save Link As" to initiate the download.

setup requires video & model weight to be at root dir

```python
virtualenv venv -p python

# activate virtual env 
# source venv\bin\activate # linux
# venv\Script\activate # windows

pip install -r requirements.txt

# cv based approach
python cv_based_approach\cv_main.py

# ml based approach
python ml_based_approach\run_inference.py

```

# Deliverables & How to submit

For the submission of this test, please fork this git repository. We would prefer a submission with a Dockerfile for building a docker image which can run your code on a standard Linux environment with docker installed (nvidia-docker will also be available when executing your code should you wish to use GPU enhancements), we will also accept code files as long as you provide clear instructions on execution and must require little to no editing from our side. You should include all necessary code files and model weights (if necessary due to filesize, upload to an external location and include a download in your Dockerfile) and a readme explaining how to execute your code in your submission. In an ideal world we will be able to pull your repository, build the docker and execute your code. Occasionally due to things out of our control this may not be the case, in this situation we may contact you to discuss your approach instead.

You should also include a report of your research into the methodologies you would take given more time (for instance assume this were a 2-3 week project) in whichever format you prefer (assuming it can be opened without specialised software). Please put this at the root of your repository with an obvious name. Your report should cover, possible techinques, pros/cons of each technique, cost of implementation and any potential issues the method could face when running in production. Please ensure to cite any references (and note that as an engineer stack overflow is an acceptable resource). This can be as simple as a raw text document as long as your thought processes are clear.

**IMPORTANT: PLEASE DO NOT INCLUDE THE ORIGINAL VIDEO IN YOUR SUBMISSION, PROVIDE INSTRUCTIONS AS TO WHERE IT SHOULD BE PLACED IN YOUR SETUP** 

# Evaluation Criteria

We have two key aims in this technical test. The first is of course to see your coding ability and the second your problem solving/research skills.

## Coding ability

We will assess your code on metrics such as:

* Code clarity (is the code easy to follow)
* Code structure (is the code easy to adapt/port/add to/maintain)
* General practices
* Success at completing task (we are not looking for perfection as you only have a few hours)

## Problem Solving/Research

We will assess your critical thinking skills on metrics such as:

* Familiarity with common libraries/frameworks
* Knowledge of standard legacy/modern computer vision methods
* End-to-end planning of solving a CV problem
* Out of the box thinking

# Troubleshooting & Support
Test@Reach.Industries

# Terms & Conditions for Interview Tests

## 1. Purpose of the Test
This test is designed to evaluate the technical prowess, problem-solving skills, and overall knowledge of the applicant. The test tasks and questions reflect challenges and scenarios the company has already addressed internally. Our goal is to gauge your approach and compare it with solutions we have already devised.

## 2. Use of Test Results
- We emphasize that the tasks presented in the tests have already been achieved by our internal team to a high standard.
- The solutions provided by candidates will purely be for evaluation purposes and will not be used in any of our products or services.
- Any resemblance between our products and a test submission is purely coincidental. We have no intention or motive to replicate your solution in our offerings.
- We will not store or retain your test results post-evaluation.

## 3. Feedback and Next Steps
Feedback may or may not be provided post-test based on company discretion. Candidates are encouraged to engage in the review meeting for a detailed understanding of their performance.

## 4. Modifications & Updates
Reach Industries reserves the right to modify these terms and conditions without prior notice. It's advised to review these T&Cs before proceeding with the test.

## 5. Consent
By participating in this test, candidates agree to these terms and conditions.
