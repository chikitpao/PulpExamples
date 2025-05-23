PuLP examples 
===
Linear programming examples using Python and library PuLP.

Library Matplotlib is also required for visualization. 


Installation
------------
You need to install Python, PuLP and Matplotlib to run my python scripts.

File list
------------
- **CircleInPolygon**
    - **CircleInPolygon.py**: Calculate circle with maximum area within a specified polygon.
    - **CircleInPolygon_Screenshot.png**: Screenshot of program CircleInPolygon.py.
- **PolycubePuzzle**
    - **PolycubePuzzle.py**: Solve polycube puzzle either with identical pieces or different pieces.
    - **PolycubePuzzleMultiple.png**: Solution plot for program PolycubePuzzle.py with option "--multiple".
    - **PolycubePuzzleSingle.png**: Solution plot for program PolycubePuzzle.py with option "--single".
- **SimpleClassifier**
    - **SimpleClassifier.py**: Calculate classifier for data with two variables.
    - **SimpleClassifier_Screenshot1.png**: Screenshot of program SimpleClassifier.py when clear classification is possible.
    - **SimpleClassifier_Screenshot2.png**: Screenshot of program SimpleClassifier.py when clear classification is not possible.
- **SudokuSolver**
    - **SudokuSolver.py**: Solve Sudoku puzzle.
    - **sudoku.txt**: Example sudoku input file.
    - **Sudoku.png**: Screenshot of program SodokuSolver.py.

Usage
------------
- CircleInPolygon/CircleInPolygon.py: Put coordinates of the polygon into the list 'points' in function 'main' and then run the program. Might not work for concave polygons at the moment.
- PolycubePuzzle/PolycubePuzzle.py: It will just print the solution of the Polycube Puzzle.
    * 1: Solving puzzle with identical pieces
        * 1.1: Change the puzzle of variable "raw_polycube" in function "main", if necessary.
        * 1.2: Run the program either without option or with option "--single".
    * 2: Solving puzzle with different pieces
        * 2.1: Change the puzzle of variable "raw_polycubes" in function "main", if necessary.
        * 2.2: Run the program with option "--multiple".
- SimpleClassifier/SimpleClassifier.py: 
    * Step 1: Put data into the list 'data' in function 'main'.
    * Step 2: Change 'x_label', 'y_label', and 'forced_buffer' to proper values.
    * Step 3: Run the program.
- SudokuSolver/SudokuSolver.py: Either change the variable "sudoku" or specify the input file name as program parameter. Put 0 for unknown numbers.


License & Copyright
-------------------
This open source release is licensed under the CC0 license. All trademarks are the property of their respective owners.
