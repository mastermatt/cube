import min2phase.search
import min2phase.tools

THREE_BY_THREE_MAX_SCRAMBLE_LENGTH = 21
THREE_BY_THREE_TIMEMIN = 200  # milliseconds
THREE_BY_THREE_TIMEOUT = 60000  # milliseconds


def generate_random_moves(first_axis_restriction=None, last_axis_restriction=None) -> str:
    """
    public PuzzleStateAndGenerator generateRandomMoves(Random r, String firstAxisRestriction, String lastAxisRestriction) {
        String randomState = Tools.randomCube(r);
        String scramble = twoPhaseSearcher.get().solution(randomState, THREE_BY_THREE_MAX_SCRAMBLE_LENGTH, THREE_BY_THREE_TIMEOUT, THREE_BY_THREE_TIMEMIN, Search.INVERSE_SOLUTION, firstAxisRestriction, lastAxisRestriction).trim();

        AlgorithmBuilder ab = new AlgorithmBuilder(this, MergingMode.CANONICALIZE_MOVES);
        try {
            ab.appendAlgorithm(scramble);
        } catch (InvalidMoveException e) {
            azzert(false, new InvalidScrambleException(scramble, e));
        }
        return ab.getStateAndGenerator();
    }
    """
    # random_state = min2phase.tools.random_cube()
    random_state = 'FBDDUFRURUUFRRFUDLDLBFFUDLFRDRRDFLBULRFDLLBBBLUURBBBLD'
    print("State:", random_state)

    scramble = min2phase.search.Search().get_solution(
        facelets=random_state,
        max_depth=THREE_BY_THREE_MAX_SCRAMBLE_LENGTH,
        time_out=THREE_BY_THREE_TIMEOUT,
        time_min=THREE_BY_THREE_TIMEMIN,
        verbose=min2phase.search.Search.INVERSE_SOLUTION,
        first_axis_restriction_str=first_axis_restriction,
        last_axis_restriction_str=last_axis_restriction,
    )

    desired_solutions = [
        "R  L  U  R2 F' R  F  L' U2 B' U2 L2 U  R2 U' L2 D  R2 U'",
        "F2 L2 U  R2 B2 D  B2 R2 U  F2 U  B' R' U' R' F' U2 R  F  D",
    ]
    print("Scramble:", scramble, "Value:", scramble in desired_solutions)

    # ab = AlgorithmBuilder(MergingMode.CANONICALIZE_MOVES);
    # ab.append_algorithm(scramble)
    # return ab.get_state_and_generator()
    return scramble


if __name__ == '__main__':
    print(generate_random_moves())
