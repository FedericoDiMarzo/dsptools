# Introduction
This growing utility library aims at incrementally covering the gap between python and matlab, in the field of audio signal processing. It will implement numerous functions related to audio signal analysis and elaboration (source separation, denoising, Weiner filtering, time and pitch scaling, reconstruction of corrupted audio), using as a backbone numpy and scipy.

Right now, dsputil is just a personal project extended to few contributors from Politecnico di Milano; in the near future, the goal of this library is reaching an adequate amount of well tested algorithms, ready to be used for research or experimentation purposes.

# Contributing
One of the main reasons behind the birth of this project is the need for well documented, properly written, and efficient algorithms in python exploiting a vector matrix formulation in numpy. Indeed, many of the resources that could be already found, lack one of those aspects, or completely miss a python implementation. 

To realize this goal, a test driven approach is employed to design and add new features to the library. Indeed before implementing any functionality, a test should be designed for it, and each valid feature should pass all the tests related to it. The definition of tests in the field of audio applications could be unclear (for example, defining a test to determine the success of a source separation algorithm, could be challenging), but it’s essential to design reliable tests to verify the correctness of the new piece of code.

If you want to offer your help for this project, and you share the same need for a high quality dsp audio oriented python library, your contribution is more than welcome. There are various way in which you could contribute for the project, shaping the future and destiny of this open source project:

- **Proposing new features**:
If you’ve got an idea for a new feature you would like to be added in the library, you could help us even just by proposing it. Indeed we think that understanding the future needs of the user is of fundamental importance.

- **Researching implementations on papers**:
The scientific literature of audio processing algorithms is vast, and choosing the best implementation could be challenging. If you feel confident in researching and giving us insights in the best solutions, your contribution could be essential.

- **Designing tests**:
Before any line of code could be written for a new functionality, a test needs to be designed. We’re currently employing a unittest methodology trying to define the boundaries of each algorithm, understanding its criteria of correctness. Your help would be useful both in the theoretical (clarifying the theory needed to design a test) and practical (implementing the test with the unittest framework, based on a theoretical base) sides.

- **Researching implementations on papers**:
The scientific literature of audio processing algorithms is vast, and choosing the best implementation could be challenging. If you feel confident in researching and giving us insights in the best solutions, your contribution could be essential.

- **Writing documentation**:
At this first prototyping phase, the algorithms are documented directly in the code with Doxygen. In order to provide their description or examples, a complete documentation will be needed soon. You could help us to write it if you wish.

- **Implementing the algorithms**:
If you’re a python developer, and you know how to properly exploit the functionalities offered by numpy and scipy, you’re the perfect candidate to contribute to develop the new algorithms in code. As already said, you should be aware of the necessity of designing algorithms that should be verified by tests, and share the same perspectives and goals of the project.

- **Improving the architecture**:
The overall architectural design of the library is of fundamental importance. If you identify possible improvements, you could propose them and they could be considered for future redesigns of the library.


# Design guidelines
If you’re willing to participate in the development of dsptools, there are few guidelines that should be followed, in order to ensure a controlled growth of the library. 

If you’re not confident in contributing to open source repositories, this video should clear your doubts:
https://www.youtube.com/watch?v=HbSjyU2vf6Y&ab_channel=TheNetNinja

If you want a crash course to unittest and test driven design, you find an useful introduction here:
https://www.youtube.com/watch?v=6tNS--WetLI&ab_channel=CoreySchafer

*(both of the authors of the videos are not related to the project)*

Here there is an incomplete list of guidelines for the project:

- **Write short, atomic and concise tests**:
The test should be divided in multiple indipendent cases, each testing a different aspect of the algorithm. If you find yourself writing longer test cases, compared to the ones already present, think again.

- **Propose a merge just after all the tests are passed**:
Before proposing a branch merging, you should design or use already designed tests and verify them all against your implementation.

- **Always propose matrix vector form solutions**:
Python for loops are inherently slow, *VERY* slow, compared to a matrix vector implementation. As a rule of thumb, never write a loop if you think it’s strictly necessary, and even if that’s the case, try to find an alternative implementation using vectors and matrices (for example, an iterative sample by sample algorithm, could be vectorized in a block form).

- **Annotate the algorithms with link and references**:
In order to understand the work of others, citing and linking the scientific papers that originated the code is essential. Try to keep track of all the references with comments.


For other information or proposals, you could write me an email:
mail.federicodimarzo@gmail.com
