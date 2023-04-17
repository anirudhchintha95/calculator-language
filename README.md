# CS 515-A - Project 2 - Calculator Language

## Submitted By

- Name: Anirudh Chintha
- Email: achintha@stevens.edu

- Name: Shubham Kulkarni
- Email: skulkarn1@stevens.edu

## Github URL

- https://github.com/anirudhchintha95/calculator-language

## Description

This project is about bulding a version of bc, a standard calculator used on the command-line in POSIX systems

## Time Spent

We spent approximately 12 hours to complete the project.

## Code Testing

We tested our program based on 3 factors and multiple test cases for each of them:

1. To check if our program is parsing the statments correctly, we added several doctests inside our Parser.
2. To make sure that the program is interpreting the parsed statments correctly based on precedence of the operators.
3. Finally, to verify the expected output is same as the generated output. We tested this with almost 100 user defined test cases to make sure we aren't facing any bugs.

Post testing the baseline functionalities, we tested four out of the seven provided extensions. Again did this step manually with several user defined steps along with making sure addition of these extensions doesn't hamper the baseline functionality.

## Bugs and Issues

Currently there are no bugs or issues in the program

## Resolved Issues or Bugs

There were several small bugs in our program:
1. Name of the variable was permitted to start with non alpha char initially.
2. Some test cases like 'print a++ - -b' was throwing a parse error since we werent verifying ++ or -- should be attached to the variable.
3. Our comments weren't functioning for certain test cases like a = /* b=5 */ 10. These were giving parse error initially.

## Extensions

1. Op-Equals
2. Relational Operations
3. Boolean Operations
4. Comments
