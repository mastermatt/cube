"""
The Coordinate Level

http://kociemba.org/math/coordlevel.htm
"""

import array
import collections
import itertools
from typing import Optional
from . import cubie_cube
from . import util

N_MOVES = 18  # The number of possible single moves. 6 faces * 3 moves per face (90°, 180°, 270°).
N_MOVES2 = 10

N_SLICE = 495  # 12 choose 4. The 4 edges from a solved slice could be in any of 12 places
N_TWIST_SYM = 324
N_FLIP_SYM = 336
N_PERM_SYM = 2768
N_M_PERM = 24
"""
 *_MOVE  = Move Table
 *_PRUNE = Pruning Table
 *_CONJUGATE  = Conjugate Table

 UD-slice = the four edges between the U-face and D-face
"""
# phase 1
UD_SLICE_MOVE: list = None  # [N_SLICE][N_MOVES]
TWIST_MOVE: list = None  # char[N_TWIST_SYM][N_MOVES]
FLIP_MOVE: list = None  # char[N_FLIP_SYM][N_MOVES]
UD_SLICE_CONJUGATE: list = None  # [N_SLICE][8]

UD_SLICE_TWIST_PRUNE: list = None  # int[N_SLICE * N_TWIST_SYM / 8 + 1]  # 20048
UD_SLICE_FLIP_PRUNE: list = None  # int[N_SLICE * N_FLIP_SYM / 8]
TWIST_FLIP_PRUNE: list = None  # int[N_FLIP_SYM * N_TWIST_SYM * 8 / 8]

# phase 2
C_PERM_MOVE: list = None  # char[N_PERM_SYM][N_MOVES]
E_PERM_MOVE: list = None  # char[N_PERM_SYM][N_MOVES2]
M_PERM_MOVE: list = None  # char[N_MPERM][N_MOVES2]
M_PERM_CONJUGATE: list = None  # char[N_MPERM][16]
MC_PERM_PRUNE: list = None  # int[N_MPERM * N_PERM_SYM / 8]
ME_PERM_PRUNE: list = None  # int[N_MPERM * N_PERM_SYM / 8]

NO = -1  # (1 << 32) - 1


def set_pruning(table: list, index: int, value: int):
    table[index >> 3] ^= (0x0f ^ value) << ((index & 7) << 2)


def get_pruning(table: list, index: int) -> int:
    """

    `index` is a dual key. The three least significant bits denote which four sequential bits
    are returned from the (assumed) 32 bit value in the table (list). The higher bits of
    `index` are the actual list index. Presumable this was only done as an efficient
    way of storing a large collection of 4 bit values.

    Returns 0 <= x < 16
    """
    # I think shifting the mask is easier to read,
    # sticking to parity for now.
    # return table[index >> 3] & 0x0f << ((index & 7) << 2)
    return (table[index >> 3] >> ((index & 7) << 2)) & 0x0f


def init_ud_slice_move_conjugate():
    """
    static void initUDSliceMoveConj() {
        CubieCube c = new CubieCube();
        CubieCube d = new CubieCube();
        for (int i=0; i<N_SLICE; i++) {
            c.setUDSlice(i);
            for (int j=0; j<N_MOVES; j+=3) {
                CubieCube.EdgeMult(c, CubieCube.moveCube[j], d);
                UDSliceMove[i][j] = (char) d.getUDSlice();
            }
            for (int j=0; j<16; j+=2) {
                CubieCube.EdgeConjugate(c, CubieCube.SymInv[j], d);
                UDSliceConj[i][j>>>1] = (char) (d.getUDSlice() & 0x1ff);
            }
        }
        for (int i=0; i<N_SLICE; i++) {
            for (int j=0; j<N_MOVES; j+=3) {
                int udslice = UDSliceMove[i][j];
                for (int k=1; k<3; k++) {
                    int cx = UDSliceMove[udslice & 0x1ff][j];
                    udslice = Util.permMult[udslice>>>9][cx>>>9]<<9|cx&0x1ff;
                    UDSliceMove[i][j+k] = (char)(udslice);
                }
            }
        }
    }

    JS
    function initUDSliceMoveConj() {
        var c, cx, d, i, j, k, udslice;
        c = new CubieCube;
        d = new CubieCube;
        for (i = 0; i < 495; ++i) {
            setComb(c.ep, i);
            for (j = 0; j < 18; j += 3) {
                EdgeMult(c, moveCube[j], d);
                UDSliceMove[i][j] = getComb(d.ep, 8);
            }
            for (j = 0; j < 16; j += 2) {
                EdgeConjugate(c, SymInv[j], d);
                UDSliceConj[i][j >>> 1] = getComb(d.ep, 8) & 511;
            }
        }
        for (i = 0; i < 495; ++i) {
            for (j = 0; j < 18; j += 3) {
                udslice = UDSliceMove[i][j];
                for (k = 1; k < 3; ++k) {
                    cx = UDSliceMove[udslice & 511][j];
                    udslice = permMult[udslice >>> 9][cx >>> 9] << 9 | cx & 511;
                    UDSliceMove[i][j + k] = udslice;
                }
            }
        }
    }
    """
    global UD_SLICE_MOVE, UD_SLICE_CONJUGATE

    c = cubie_cube.CubieCube()
    d = cubie_cube.CubieCube()

    inner = [0] * N_MOVES
    UD_SLICE_MOVE = [inner[:] for _ in range(N_SLICE)]

    inner = [0] * 8
    UD_SLICE_CONJUGATE = [inner[:] for _ in range(N_SLICE)]

    for i in range(N_SLICE):
        c.set_ud_slice(i)

        for j in range(0, N_MOVES, 3):
            cubie_cube.edge_multiply(c, cubie_cube.MOVE_CUBE[j], d)
            UD_SLICE_MOVE[i][j] = d.get_ud_slice()

        for j in range(0, 16, 2):
            cubie_cube.edge_conjugate(c, cubie_cube.SYM_INV[j], d)
            UD_SLICE_CONJUGATE[i][j >> 1] = d.get_ud_slice() & 0x1ff

    for i, j in itertools.product(range(N_SLICE), range(0, N_MOVES, 3)):
        ud_slice = UD_SLICE_MOVE[i][j]
        for k in range(1, 3):
            cx = UD_SLICE_MOVE[ud_slice & 0x1ff][j]
            ud_slice = util.PERM_MULT[ud_slice >> 9][cx >> 9] << 9 | cx & 0x1ff
            UD_SLICE_MOVE[i][j + k] = ud_slice


def init_flip_move():
    """
    static void initFlipMove() {
        CubieCube c = new CubieCube();
        CubieCube d = new CubieCube();
        for (int i=0; i<N_FLIP_SYM; i++) {
            c.setFlip(CubieCube.FlipS2R[i]);
            for (int j=0; j<N_MOVES; j++) {
                CubieCube.EdgeMult(c, CubieCube.moveCube[j], d);
                FlipMove[i][j] = (char) d.getFlipSym();
            }
        }
    }
    """
    global FLIP_MOVE

    c = cubie_cube.CubieCube()
    d = cubie_cube.CubieCube()

    inner = array.ArrayType('h', [0] * N_MOVES)
    FLIP_MOVE = [inner[:] for _ in range(N_FLIP_SYM)]

    for i in range(N_FLIP_SYM):
        c.set_flip(cubie_cube.FLIP_S2R[i])
        for j in range(N_MOVES):
            cubie_cube.edge_multiply(c, cubie_cube.MOVE_CUBE[j], d)
            FLIP_MOVE[i][j] = d.get_flip_sym()


def init_twist_move():
    """
    static void initTwistMove() {
        CubieCube c = new CubieCube();
        CubieCube d = new CubieCube();
        for (int i=0; i<N_TWIST_SYM; i++) {
            c.setTwist(CubieCube.TwistS2R[i]);
            for (int j=0; j<N_MOVES; j++) {
                CubieCube.CornMult(c, CubieCube.moveCube[j], d);
                TwistMove[i][j] = (char) d.getTwistSym();
            }
        }
    }
    """
    global TWIST_MOVE

    c = cubie_cube.CubieCube()
    d = cubie_cube.CubieCube()

    inner = array.ArrayType('h', [0] * N_MOVES)
    TWIST_MOVE = [inner[:] for _ in range(N_TWIST_SYM)]

    for i in range(N_TWIST_SYM):
        c.set_twist(cubie_cube.TWIST_S2R[i])
        for j in range(N_MOVES):
            cubie_cube.corner_multiply(c, cubie_cube.MOVE_CUBE[j], d)
            TWIST_MOVE[i][j] = d.get_twist_sym()


def init_c_perm_move():
    """
    static void initCPermMove() {
        CubieCube c = new CubieCube();
        CubieCube d = new CubieCube();
        for (int i=0; i<N_PERM_SYM; i++) {
            c.setCPerm(CubieCube.EPermS2R[i]);
            for (int j=0; j<N_MOVES; j++) {
                CubieCube.CornMult(c, CubieCube.moveCube[j], d);
                CPermMove[i][j] = (char) d.getCPermSym();
            }
        }
    }
    """
    global C_PERM_MOVE

    c = cubie_cube.CubieCube()
    d = cubie_cube.CubieCube()

    inner = array.ArrayType('l', [0] * N_MOVES)
    C_PERM_MOVE = [inner[:] for _ in range(N_PERM_SYM)]

    for i in range(N_PERM_SYM):
        c.set_c_perm(cubie_cube.E_PERM_S2R[i])
        for j in range(N_MOVES):
            cubie_cube.corner_multiply(c, cubie_cube.MOVE_CUBE[j], d)
            C_PERM_MOVE[i][j] = d.get_c_perm_sym()


def init_e_perm_move():
    """
    static char[][] EPermMove = new char[N_PERM_SYM][N_MOVES2];
    static void initEPermMove() {
        CubieCube c = new CubieCube();
        CubieCube d = new CubieCube();
        for (int i=0; i<N_PERM_SYM; i++) {
            c.setEPerm(CubieCube.EPermS2R[i]);
            for (int j=0; j<N_MOVES2; j++) {
                CubieCube.EdgeMult(c, CubieCube.moveCube[Util.ud2std[j]], d);
                EPermMove[i][j] = (char) d.getEPermSym();
            }
        }
    }
    """
    global E_PERM_MOVE

    c = cubie_cube.CubieCube()
    d = cubie_cube.CubieCube()

    inner = array.ArrayType('l', [0] * N_MOVES2)
    E_PERM_MOVE = [inner[:] for _ in range(N_PERM_SYM)]

    for i in range(N_PERM_SYM):
        c.set_e_perm(cubie_cube.E_PERM_S2R[i])
        for j in range(N_MOVES2):
            cubie_cube.edge_multiply(c, cubie_cube.MOVE_CUBE[util.UD_2_STD[j]], d)
            E_PERM_MOVE[i][j] = d.get_e_perm_sym()


def init_m_perm_move_conjugate():
    """
    static void initMPermMoveConj() {
        CubieCube c = new CubieCube();
        CubieCube d = new CubieCube();
        for (int i=0; i<N_MPERM; i++) {
            c.setMPerm(i);
            for (int j=0; j<N_MOVES2; j++) {
                CubieCube.EdgeMult(c, CubieCube.moveCube[Util.ud2std[j]], d);
                MPermMove[i][j] = (char) d.getMPerm();
            }
            for (int j=0; j<16; j++) {
                CubieCube.EdgeConjugate(c, CubieCube.SymInv[j], d);
                MPermConj[i][j] = (char) d.getMPerm();
            }
        }
    }
    """
    global M_PERM_MOVE, M_PERM_CONJUGATE

    c = cubie_cube.CubieCube()
    d = cubie_cube.CubieCube()

    inner = array.ArrayType('h', (0 for _ in range(N_MOVES2)))
    M_PERM_MOVE = [inner[:] for _ in range(N_M_PERM)]

    inner = array.ArrayType('h', (0 for _ in range(16)))
    M_PERM_CONJUGATE = [inner[:] for _ in range(N_M_PERM)]

    for i in range(N_M_PERM):
        c.set_m_perm(i)

        for j in range(N_MOVES2):
            cubie_cube.edge_multiply(c, cubie_cube.MOVE_CUBE[util.UD_2_STD[j]], d)
            M_PERM_MOVE[i][j] = d.get_m_perm()

        for j in range(16):
            cubie_cube.edge_conjugate(c, cubie_cube.SYM_INV[j], d)
            M_PERM_CONJUGATE[i][j] = d.get_m_perm()


def init_twist_flip_prune():
    global TWIST_FLIP_PRUNE

    depth = 0
    done = 8
    n_size = N_FLIP_SYM * N_TWIST_SYM * 8

    TWIST_FLIP_PRUNE = array.ArrayType('l', (-1 for _ in range(N_FLIP_SYM * N_TWIST_SYM)))

    for i in range(8):
        set_pruning(TWIST_FLIP_PRUNE, i, 0)

    while done < n_size:
        inv = depth > 6
        select, check = (0x0f, depth) if inv else (depth, 0x0f)
        depth += 1

        for i in range(n_size):
            if get_pruning(TWIST_FLIP_PRUNE, i) != select:
                continue

            twist = i // 2688
            flip = i % 2688
            f_sym = i & 7
            flip >>= 3

            for m in range(N_MOVES):
                twist_x = TWIST_MOVE[twist][m]
                t_sym_x = twist_x & 7
                twist_x >>= 3
                flip_x = FLIP_MOVE[flip][cubie_cube.SYM_8_MOVE[f_sym][m]]
                sym_multiplier = cubie_cube.SYM_8_MULT[flip_x & 7][f_sym]
                f_sym_x = cubie_cube.SYM_8_MULT_INV[sym_multiplier][t_sym_x]

                flip_x >>= 3
                idx = (twist_x * 336 + flip_x) << 3 | f_sym_x

                if get_pruning(TWIST_FLIP_PRUNE, idx) != check:
                    continue

                done += 1

                if inv:
                    set_pruning(TWIST_FLIP_PRUNE, i, depth)
                    break

                set_pruning(TWIST_FLIP_PRUNE, idx, depth)
                sym = cubie_cube.SYM_STATE_TWIST[twist_x]
                sym_f = cubie_cube.SYM_STATE_FLIP[flip_x]

                if not (sym != 1 or sym_f != 1):
                    continue

                for j in range(8):
                    skip = sym_f & 1 == 1
                    sym_f >>= 1

                    if skip:
                        continue

                    f_sym_xx = cubie_cube.SYM_8_MULT_INV[f_sym_x][j]

                    for k in range(8):
                        if (sym & (1 << k)) == 0:
                            continue

                        idx_x = twist_x * 2688
                        idx_x += flip_x << 3 | cubie_cube.SYM_8_MULT_INV[f_sym_xx][k]
                        if get_pruning(TWIST_FLIP_PRUNE, idx_x) == 0x0f:
                            set_pruning(TWIST_FLIP_PRUNE, idx_x, depth)
                            done += 1


def _init_raw_sym_prune(
    prune_table: list,
    inv_depth: int,
    raw_move: list,
    raw_conj: list,
    sym_move: list,
    sym_state: list,
    sym_switch: Optional[list],
    move_map: Optional[list],
    sym_shift: int,
):

    sym_mask = (1 << sym_shift) - 1
    n_raw = len(raw_move)
    n_sym = len(sym_move)
    n_size = n_raw * n_sym
    n_moves = len(raw_move[0])

    # if len(prune_table) != ((n_size+7)//8):
    #     raise ValueError(len(prune_table), ((n_size+7)//8))
    # for i in range((n_size+7)//8):
    #     prune_table[i] = -1

    set_pruning(prune_table, 0, 0)

    depth = 0
    done = 1

    while done < n_size:  # loops 9x

        inv = depth > inv_depth
        select, check = (0x0f, depth) if inv else (depth, 0x0f)
        depth += 1
        assert depth < 15
        i = 0

        # loops [0, 20068, 20137, 20504, 23722, 45361, 104841, 113748, 136809, 160229, 0, 0]
        while i < n_size:
            val = prune_table[i >> 3]

            if not inv and val == NO:
                i += 8
                continue

            for _ in range(i, min(i + 8, n_size)):
                is_select = get_pruning(prune_table, i) == select
                i += 1

                if not is_select:
                    break

                # raw = i % n_raw
                raw = (i - 1) % n_raw
                # sym = i // n_raw
                sym = (i - 1) // n_raw
                for m in range(n_moves):
                    sym_x = sym_move[sym][move_map[m] if move_map else m]
                    # sym_x_cache = sym_x
                    mmm = sym_x & sym_mask
                    raw_x = raw_conj[raw_move[raw][m] & 0x1ff][mmm]

                    if sym_x < 0:
                        raise ValueError(sym_x)
                    sym_x >>= sym_shift
                    idx = sym_x * n_raw + raw_x
                    x = get_pruning(prune_table, idx)
                    if x != check:
                        continue

                    done += 1

                    if inv:
                        set_pruning(prune_table, i - 1, depth)
                        break

                    set_pruning(prune_table, idx, depth)
                    sym_state_val = sym_state[sym_x]
                    for j in itertools.count(1):
                        sym_state_val >>= 1
                        if sym_state_val == 0:
                            break

                        if sym_state_val & 1 == 0:
                            continue

                        sym_s = j ^ (sym_switch[j] if sym_switch else 0)
                        ix = sym_x * n_raw + raw_conj[raw_x][sym_s]
                        # if get_pruning(prune_table, ix) == 0x0f:
                        z = get_pruning(prune_table, ix)
                        if z == 0x0f:
                            set_pruning(prune_table, ix, depth)
                            done += 1


def _init_raw_sym_prune2(
    prune_table: list,
    raw_move: list,
    raw_conj: list,
    sym_move: list,
    sym_state: list,
    sym_shift: int,
    sym_switch: list = None,
    move_map: list = None,
):

    sym_mask = (1 << sym_shift) - 1
    n_raw = len(raw_move)
    n_sym = len(sym_move)
    n_size = n_raw * n_sym
    n_moves = len(raw_move[0])

    print(n_size, n_moves)

    # if len(prune_table) != ((n_size+7)//8):
    #     raise ValueError(len(prune_table), ((n_size+7)//8))
    # for i in range((n_size+7)//8):
    #     prune_table[i] = -1

    # prune_table = array.ArrayType('l', [NO] * ((n_size + 7) // 8))
    set_pruning(prune_table, 0, 0)
    queue = collections.deque([0], n_size)
    unset_value: int = 0x0f
    complexity = 0
    done = 1

    while done < n_size:
        current_idx = queue.popleft()
        raw = current_idx % n_raw
        sym = current_idx // n_raw

        for m in range(n_moves):
            complexity += 1
            sym_x = sym_move[sym][move_map[m] if move_map else m]
            raw_x = raw_conj[raw_move[raw][m] & 0x1ff][sym_x & sym_mask]

            sym_x >>= sym_shift
            idx = sym_x * n_raw + raw_x

            x = get_pruning(prune_table, idx)
            if x != unset_value:
                continue

            depth: int = get_pruning(prune_table, current_idx)
            set_pruning(prune_table, idx, depth + 1)
            queue.append(idx)
            done += 1

            sym_state_val = sym_state[sym_x]
            for j in itertools.count(1):
                sym_state_val >>= 1
                if sym_state_val == 0:
                    break

                if sym_state_val & 1 == 0:
                    continue

                sym_s = (j ^ sym_switch[j]) if sym_switch else j
                ix = sym_x * n_raw + raw_conj[raw_x][sym_s]
                z = get_pruning(prune_table, ix)
                if z == unset_value:
                    set_pruning(prune_table, ix, depth+1)
                    queue.append(ix)
                    done += 1

    # return prune_table
    print(complexity)


def _init_raw_sym_prune3(
    prune_table: list,
    raw_move: list,
    raw_conj: list,
    sym_move: list,
    sym_state: list,
    sym_shift: int,
    sym_switch: list = None,
    move_map: list = None,
):

    sym_mask = (1 << sym_shift) - 1
    n_raw = len(raw_move)
    n_sym = len(sym_move)
    n_size = n_raw * n_sym
    n_moves = len(raw_move[0])

    print(n_size, n_moves)

    # prune_table = array.ArrayType('l', [NO] * ((n_size + 7) // 8))
    prune_table[0] = 0
    queue = collections.deque([0])
    # unset_value: int = 0x0f
    unset_value: int = NO
    complexity = 0
    done = 1

    while done < n_size:
        current_idx = queue.popleft()
        raw = current_idx % n_raw
        sym = current_idx // n_raw

        for m in range(n_moves):
            sym_x = sym_move[sym][move_map[m] if move_map else m]
            raw_x = raw_conj[raw_move[raw][m] & 0x1ff][sym_x & sym_mask]
            sym_x >>= sym_shift
            idx = sym_x * n_raw + raw_x

            # x: int = prune_table[idx]
            # x = get_pruning(prune_table, idx)
            # if x != unset_value:
            #     continue
            complexity += 1
            if prune_table[idx] != unset_value:
                continue

            # depth: int = get_pruning(prune_table, current_idx)
            # set_pruning(prune_table, idx, depth + 1)
            depth = prune_table[current_idx]
            prune_table[idx] = depth + 1
            queue.append(idx)
            done += 1

            sym_state_val = sym_state[sym_x]
            for j in itertools.count(1):
                sym_state_val >>= 1
                if sym_state_val == 0:
                    break

                if sym_state_val & 1 == 0:
                    continue

                sym_s = (j ^ sym_switch[j]) if sym_switch else j
                ix = sym_x * n_raw + raw_conj[raw_x][sym_s]
                # z = prune_table[ix]
                # if z == unset_value:
                if prune_table[ix] == unset_value:
                    prune_table[ix] = depth+1
                    queue.append(ix)
                    done += 1

    print(complexity)


def set_basic_pruning(table, index, value):
    if table[index] is None:
        table[index] = value
    else:
        print("index already set", index, value)
        table[index] ^= value


def get_basic_pruning(table, index):
    return table[index]


# import time
def init_slice_twist_prune():
    global UD_SLICE_TWIST_PRUNE

    # start = time.time()
    UD_SLICE_TWIST_PRUNE = array.ArrayType('l', [NO] * (N_SLICE * N_TWIST_SYM // 8 + 1))
    _init_raw_sym_prune(
        prune_table=UD_SLICE_TWIST_PRUNE,
        inv_depth=6,
        raw_move=UD_SLICE_MOVE,
        raw_conj=UD_SLICE_CONJUGATE,
        sym_move=TWIST_MOVE,
        sym_state=cubie_cube.SYM_STATE_TWIST,
        sym_switch=None,
        move_map=None,
        sym_shift=3,
    )
    # print("existing way", time.time() - start)
    #
    # start = time.time()
    # test = array.ArrayType('l', [NO] * (N_SLICE * N_TWIST_SYM // 8 + 1))
    # _init_raw_sym_prune2(
    #     prune_table=test,
    #     raw_move=UD_SLICE_MOVE,
    #     raw_conj=UD_SLICE_CONJUGATE,
    #     sym_move=TWIST_MOVE,
    #     sym_state=cubie_cube.SYM_STATE_TWIST,
    #     sym_shift=3,
    # )
    # print("py bfs way", time.time() - start)
    # print(len(test), test)
    #
    # start = time.time()
    # test = array.ArrayType('l', [NO] * (N_SLICE * N_TWIST_SYM))
    # _init_raw_sym_prune3(
    #     prune_table=test,
    #     raw_move=UD_SLICE_MOVE,
    #     raw_conj=UD_SLICE_CONJUGATE,
    #     sym_move=TWIST_MOVE,
    #     sym_state=cubie_cube.SYM_STATE_TWIST,
    #     sym_shift=3,
    # )
    # print("py bfs way 2", time.time() - start)
    # print(len(test), test)
    #
    # a = [
    #     1 if i == j else 0
    #     for i, j in zip(UD_SLICE_TWIST_PRUNE, test)
    # ]
    #
    # raise ValueError(UD_SLICE_TWIST_PRUNE == test,
    #                  len(UD_SLICE_TWIST_PRUNE), len(test), len(a), sum(a))


def init_slice_flip_prune():
    global UD_SLICE_FLIP_PRUNE

    UD_SLICE_FLIP_PRUNE = array.ArrayType('l', [NO] * (N_SLICE * N_FLIP_SYM // 8 + 1))
    _init_raw_sym_prune(
        prune_table=UD_SLICE_FLIP_PRUNE,
        inv_depth=6,
        raw_move=UD_SLICE_MOVE,
        raw_conj=UD_SLICE_CONJUGATE,
        sym_move=FLIP_MOVE,
        sym_state=cubie_cube.SYM_STATE_FLIP,
        sym_switch=None,
        move_map=None,
        sym_shift=3,
    )


def init_m_e_perm_prune():
    global ME_PERM_PRUNE

    ME_PERM_PRUNE = array.ArrayType('l', [NO] * (N_M_PERM * N_PERM_SYM // 8))
    _init_raw_sym_prune(
        prune_table=ME_PERM_PRUNE,
        inv_depth=7,
        raw_move=M_PERM_MOVE,
        raw_conj=M_PERM_CONJUGATE,
        sym_move=E_PERM_MOVE,
        sym_state=cubie_cube.SYM_STATE_PERM,
        sym_switch=None,
        move_map=None,
        sym_shift=4,
    )


def init_m_c_perm_prune():
    global MC_PERM_PRUNE

    MC_PERM_PRUNE = array.ArrayType('l', [NO] * (N_M_PERM * N_PERM_SYM // 8))
    _init_raw_sym_prune(
        prune_table=MC_PERM_PRUNE,
        inv_depth=10,
        raw_move=M_PERM_MOVE,
        raw_conj=M_PERM_CONJUGATE,
        sym_move=C_PERM_MOVE,
        sym_state=cubie_cube.SYM_STATE_PERM,
        sym_switch=cubie_cube.E2C,
        move_map=util.UD_2_STD,
        sym_shift=4,
    )
