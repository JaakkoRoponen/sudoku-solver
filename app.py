from flask import abort, Flask, jsonify, render_template, request
import numpy as np

from sudoku import Sudoku


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/create')
def create():
    sudoku = Sudoku()
    puzzle = sudoku.create_puzzle()
    return jsonify(puzzle.tolist())


@app.route('/random', methods=['POST'])
def random():
    r = request.get_json()
    try:
        puzzle = np.array(r['puzzle']).reshape((9, 9))
    except ValueError:
        abort(400)

    sudoku = Sudoku()
    sudoku.solve(puzzle, solve=r['solve'])
    number = sudoku.get_random_number(puzzle, r['row'], r['col'])

    return jsonify(int(number))


@app.route('/solver', methods=['POST'])
def solver():
    puzzle = request.get_json()
    try:
        puzzle = np.array(puzzle).reshape((9, 9))
    except ValueError:
        abort(400)

    sudoku = Sudoku()
    solved = sudoku.solve(puzzle)

    deduced = solved - puzzle  # remove non-deduced numbers

    return jsonify(deduced.tolist())
