# doc.tex

doc.tex should not be touched except for addition of new \input commands in case of anything to be added on the appendix.

All actual modifications should be in the section files in the sects folder.

# Text Format

Every line should have it's own separate line. So instead of a single long line, like right now, things should have their own lines.

Here is an example.
This is how it should be.
This is because git works better with lines.
Thus, by separating sentences per line,
and even breaking at logical points,
we can very easily compare earlier and later versions of the text.

# Sections

Sections already have their own \section command and their titles.

The structure should be as follows

\section{Section Name}
--empty line--
Text
Text
--empty line to start new paragraph
Text
Text


...
etc

# How to build

Running the command "make" in a linux environment with the texlive distribution and make installed will automatically build the file in the publish directory. Warning, the publish directory is outside of the tex directory, at the root of the git repository.

Neither the temporary .build directory nor the publish directory are to be committed to the repository. They are in the .gitignore.

The command "make clean" will delete the .build and publish repository. This is useful in case for some reason make is not working properly, and you want to reset the environment.
