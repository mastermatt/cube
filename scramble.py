
import random

import min2phase.cubie_cube
import min2phase.search
import min2phase.tools
import min2phase.util

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
    # random_state = 'FBDDUFRURUUFRRFUDLDLBFFUDLFRDRRDFLBULRFDLLBBBLUURBBBLD'
    random_state = 'UUDUUDUUDRRRRRRBBBRRBFFBFFLDDUDDUDDUFFFLLLLLLFLLFBBRBB'  # U R2

    # random_moves = ' '.join([
    #     min2phase.util.MOVE_2_STR[i]
    #     for i in random.choices(min2phase.util.UD_2_STD, k=20)
    # ])
    # print(random_moves)

    cc = min2phase.cubie_cube.CubieCube()
    # cc.apply_str_moves(  "U R2 U U U L2 R2 D R2 D F2 D")
    # cc.apply_str_moves("U R2 U U U L2 R2 D R2 D F2 D F2 D")
    cc.apply_str_moves("R D L ")
    # cc.apply_str_moves(random_moves)
    random_state = str(cc)

    random_state_formatted = ''.join(
        '[{}]'.format(
            ' '.join(random_state[i+j:i+j+3] for j in range(0, 9, 3))
        ) for i in range(0, 54, 9)
    )
    print("State:", random_state_formatted)
    scramble = min2phase.search.Search().get_solution(
        facelets=random_state,
        max_depth=THREE_BY_THREE_MAX_SCRAMBLE_LENGTH,
        time_out=THREE_BY_THREE_TIMEOUT,
        time_min=THREE_BY_THREE_TIMEMIN,
        inverse_solution=False,
        append_length=True,
        use_separator=True,
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
    generate_random_moves()
