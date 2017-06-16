import random
import time
from . import util
from . import coord_cube
from . import cubie_cube

USE_TWIST_FLIP_PRUNE = False
STATE_RANDOM = list()
STATE_SOLVED = list()

INITED = False


def init_index():
    global INITED

    if INITED:
        return

    print("Initializing the indexes")
    start = time.time()

    cubie_cube.init_move()
    print('init_move', time.time()-start)
    cubie_cube.init_sym()
    print('init_sym', time.time()-start)
    cubie_cube.init_flip_sym_2_raw()
    print('init_flip_sym_2_raw', time.time()-start)
    cubie_cube.init_twist_sym_2_raw()
    print('init_twist_sym_2_raw', time.time()-start)

    cubie_cube.init_perm_sym_2_raw()
    print('init_perm_sym_2_raw', time.time()-start)
    coord_cube.init_flip_move()
    print('init_flip_move', time.time()-start)
    coord_cube.init_twist_move()
    print('init_twist_move', time.time()-start)
    coord_cube.init_ud_slice_move_conjugate()
    print('init_ud_slice_move_conjugate', time.time()-start)

    coord_cube.init_c_perm_move()
    print('init_c_perm_move', time.time()-start)
    coord_cube.init_e_perm_move()
    print('init_e_perm_move', time.time()-start)
    coord_cube.init_m_perm_move_conjugate()
    print('init_m_perm_move_conjugate', time.time()-start)

    if USE_TWIST_FLIP_PRUNE:
        coord_cube.init_twist_flip_prune()
        print('')
        print('init_twist_flip_prune', time.time()-start)

    coord_cube.init_slice_twist_prune()
    print('init_slice_twist_prune', time.time()-start)

    coord_cube.init_slice_flip_prune()
    print('init_slice_flip_prune', time.time()-start)

    coord_cube.init_m_e_perm_prune()
    print('init_m_e_perm_prune', time.time()-start)

    coord_cube.init_m_c_perm_prune()
    print('init_m_c_perm_prune', time.time()-start)

    end = time.time()
    print("Done.", end-start)

    INITED = True


def random_cube():
    return random_state(STATE_RANDOM, STATE_RANDOM, STATE_RANDOM, STATE_RANDOM)


def random_state(cp: list, co: list, ep: list, eo: list):
    """
    protected static String randomState(byte[] cp, byte[] co, byte[] ep, byte[] eo, Random gen) {
        int parity;
        int cntUE = ep == STATE_RANDOM ? 12 : countUnknown(ep);
        int cntUC = cp == STATE_RANDOM ? 8 : countUnknown(cp);
        int cpVal, epVal;
        if (cntUE < 2) {	//ep != STATE_RANDOM
            if (ep == STATE_SOLVED) {
                epVal = parity = 0;
            } else {
                parity = resolvePerm(ep, cntUE, -1, gen);
                epVal = Util.getNPerm(ep, 12);
            }
            if (cp == STATE_SOLVED) {
                cpVal = 0;
            } else if (cp == STATE_RANDOM) {
                do {
                    cpVal = gen.nextInt(40320);
                } while (Util.getNParity(cpVal, 8) != parity);
            } else {
                resolvePerm(cp, cntUC, parity, gen);
                cpVal = Util.getNPerm(cp, 8);
            }
        } else {	//ep != STATE_SOLVED
            if (cp == STATE_SOLVED) {
                cpVal = parity = 0;
            } else if (cp == STATE_RANDOM) {
                cpVal = gen.nextInt(40320);
                parity = Util.getNParity(cpVal, 8);
            } else {
                parity = resolvePerm(cp, cntUC, -1, gen);
                cpVal = Util.getNPerm(cp, 8);
            }
            if (ep == STATE_RANDOM) {
                do {
                    epVal = gen.nextInt(479001600);
                } while (Util.getNParity(epVal, 12) != parity);
            } else {
                resolvePerm(ep, cntUE, parity, gen);
                epVal = Util.getNPerm(ep, 12);
            }
        }
        return Util.toFaceCube(new CubieCube(
            cpVal,
            co == STATE_RANDOM ? gen.nextInt(2187) :
                    (co == STATE_SOLVED ? 0 : resolveOri(co, 3, gen)),
            epVal,
            eo == STATE_RANDOM ? gen.nextInt(2048) :
                    (eo == STATE_SOLVED ? 0 : resolveOri(eo, 2, gen))
        ));
    }
    """
    cnt_ue = 12 if ep is STATE_RANDOM else _count_unknown(ep)
    cnt_uc = 8 if cp is STATE_RANDOM else _count_unknown(cp)

    if cnt_ue < 2:  # ep != STATE_RANDOM
        if ep is STATE_SOLVED:
            ep_val = parity = 0
        else:
            parity = _resolve_perm(ep, cnt_ue, -1)
            ep_val = util.get_n_perm(ep, 12)

        if cp is STATE_SOLVED:
            cp_val = 0
        elif cp == STATE_RANDOM:
            while True:
                cp_val = random.randrange(40320)
                if util.get_n_parity(cp_val, 8) == parity:
                    break
        else:
            _resolve_perm(cp, cnt_uc, parity)
            cp_val = util.get_n_perm(cp, 8)

    else:  # ep != STATE_SOLVED
        if ep is STATE_SOLVED:
            cp_val = parity = 0
        elif cp is STATE_RANDOM:
            cp_val = random.randrange(40320)
            parity = util.get_n_parity(cp_val, 8)
        else:
            parity = _resolve_perm(ep, cnt_uc, -1)
            cp_val = util.get_n_perm(cp, 8)

        if ep is STATE_RANDOM:
            while True:
                ep_val = random.randrange(479001600)
                if util.get_n_parity(ep_val, 12) == parity:
                    break
        else:
            _resolve_perm(ep, cnt_ue, parity)
            ep_val = util.get_n_perm(ep, 12)

    if co is STATE_RANDOM:
        co_val = random.randrange(2187)
    elif co is STATE_SOLVED:
        co_val = 0
    else:
        print("resolving orientation", co)
        co_val = _resolve_orientation(co, 3)

    if eo is STATE_RANDOM:
        eo_val = random.randrange(2048)
    elif eo is STATE_SOLVED:
        eo_val = 0
    else:
        print("resolving orientation", eo)
        eo_val = _resolve_orientation(eo, 2)

    print(dict(
        c_perm=cp_val,
        twist=co_val,
        e_perm=ep_val,
        flip=eo_val,
    ))

    cc = cubie_cube.CubieCube(
        c_perm=cp_val,
        twist=co_val,
        e_perm=ep_val,
        flip=eo_val,
    )
    print(cc)
    return util.to_face_cube(cc)


def _count_unknown(arr: list):
    if arr is STATE_SOLVED:
        return 0

    return len([i for i in arr if i == -1])


def _resolve_perm(arr: list, cnt_u: int, parity: int):
    if arr is STATE_SOLVED:
        return 0

    if arr is STATE_RANDOM:
        return random.randrange(2) if parity == -1 else parity

    val = list(range(12))
    for v in arr:
        if v != -1:
            val[v] = -1

    idx = 0
    for i, v in enumerate(arr):
        if val[i] != -1:
            idx += 1
            j = random.randrange(idx)
            val[idx], val[j] = val[j], val[i]

    last = -1
    for i, v in enumerate(arr):
        if cnt_u <= 0:
            break
        if v == -1:
            if cnt_u == 2:
                last = i
            cnt_u -= 1
            arr[i] = val[cnt_u]

    p = util.get_n_parity(util.get_n_perm(arr, len(arr)), len(arr))
    if p == 1-parity and last != -1:
        arr[idx-1], arr[last] = arr[last], arr[idx-1]
    return p


def _resolve_orientation(arr: list, base: int) -> int:
    """
    private static int resolveOri(byte[] arr, int base, Random gen) {
        int sum = 0, idx = 0, lastUnknown = -1;
        for (int i=0; i<arr.length; i++) {
            if (arr[i] == -1) {
                arr[i] = (byte) gen.nextInt(base);
                lastUnknown = i;
            }
            sum += arr[i];
        }
        if (sum % base != 0 && lastUnknown != -1) {
            arr[lastUnknown] = (byte) ((30 + arr[lastUnknown] - sum) % base);
        }
        for (int i=0; i<arr.length-1; i++) {
            idx *= base;
            idx += arr[i];
        }
        return idx;
    }
    :return:
    """
    total = 0
    idx = 0
    last_unknown = -1
    for i, v in enumerate(arr):
        if v == -1:
            arr[i] = random.randrange(base)
            last_unknown = i
        total += arr[i]

    if total & base != 0 and last_unknown != -1:
        arr[last_unknown] = ((30 + arr[last_unknown] - total) % base)

    for v in arr[:-1]:
        idx *= base
        idx += v

    return idx


if __name__ == '__main__':
    init_index()
