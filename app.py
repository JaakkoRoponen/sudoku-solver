from flask import abort, Flask, jsonify, render_template, request
import numpy as np

from sudoku import Sudoku


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solver', methods=['POST'])
def solver():
    puzzle = request.get_json()
    try:
        puzzle = np.array(puzzle).reshape((9, 9))
    except ValueError:
        abort(400)

    sudoku = Sudoku()
    solved = sudoku.solve(puzzle)

    puzzle = solved - puzzle # Gets deduced numbers

    return jsonify(puzzle.tolist())
