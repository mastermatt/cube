from typing import List
import array
import itertools
from . import util

# ClassIndexToRepresentantArrays
FLIP_S2R: list = None  # array.ArrayType('u', 'x' * 336)
TWIST_S2R: list = None  # array.ArrayType('u', 'x' * 324)
E_PERM_S2R: list = None  # array.ArrayType('u', 'x' * 2768)

# Notice that Edge Perm Coordinate and Corner Perm Coordinate are the same symmetry structure.
# So their ClassIndexToRepresentantArray are the same.
# And when x is RawEdgePermCoordinate, y*16+k is SymEdgePermCoordinate, y*16+(k^e2c[k]) will
# be the SymCornerPermCoordinate of the State whose RawCornerPermCoordinate is x.
E2C = [0, 0, 0, 0, 1, 3, 1, 3, 1, 3, 1, 3, 0, 0, 0, 0]

M2E_PERM: list = None  # array.ArrayType('u', 'x' * 40320)

# Raw-Coordinate to Sym-Coordinate, only for speeding up initialization.
FLIP_R2S: list = None  # array.ArrayType('u', 'x' * 2187)
TWIST_R2S: list = None  # char[2187]
E_PERM_R2S: list = None

SYM_STATE_TWIST: list = None
SYM_STATE_FLIP: list = None  # array.ArrayType('u', 'x' * 336)
SYM_STATE_PERM: list = None  # array.ArrayType('u', 'x' * 2768)


class CubieCube(object):
    """
    Stores the scrambled state of a 3x3x3 puzzle cube.

    In order to denote all the possible combinations of scrambles the
    54 cubies could be in requires four lists:

    - corner positions (cp) has the relative position of each of the
        eight corners. Each digit in the list denotes their relative
        positions when in a solved state while the index of each digit
        in the list denotes the current relative position for each corner.

    - corner orientations (co) are the twists of each corner. Each
        corner represents it's twist by having a 0 (not twisted),
        1 (twisted around clockwise), or a 2 (twisted counterclockwise).

    - edge positions (ep) has the relative position of each of the
        twelve edges on the cube. Like corner positions, each digit
        and index denotes relative positions of their current state
        versus when in a solved state.

    - edge orientations (eo) keeps track how each two sided edge is flipped.
        Each of the twelve items in the list should be either a
        0 (not flipped) or a 1 (flipped).

    A solved state would look like
    self.cp = [0, 1, 2, 3, 4, 5, 6, 7]
    self.co = [0, 0, 0, 0, 0, 0, 0, 0]
    self.ep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    self.eo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """

    def __init__(self, c_perm=0, twist=0, e_perm=0, flip=0):
        """
        Without params, the cube defaults to a solved state.
        """
        if isinstance(c_perm, CubieCube):
            # allow passing in a cubie cube instance to create a copy
            self.copy(c_perm)
            return

        # only set these to None before really setting to make the
        # linter happy and helps with documenting the code.
        # self.cp = None  # corner positions
        # self.co = None  # corner orientations. 0, 1, or 2 depending on corner twist
        # self.ep = None  # edge positions
        # self.eo = None  # edge orientations. 0 or 1 if flipped
        self.cp = [0, 1, 2, 3, 4, 5, 6, 7]
        self.co = [0, 0, 0, 0, 0, 0, 0, 0]
        self.ep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.eo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if c_perm:
            self.set_c_perm(c_perm)

        if twist:
            self.set_twist(twist)

        if e_perm:
            util.set_n_perm(self.ep, e_perm, 12)  # why not call set_e_perm?

        if flip:
            self.set_flip(flip)

    def copy(self, other):
        self.cp = other.cp[:]
        self.co = other.co[:]
        self.ep = other.ep[:]
        self.eo = other.eo[:]

    def __str__(self):
        return util.to_face_cube(self)

    def __repr__(self):
        return str({
            'cp': self.cp,
            'co': self.co,
            'ep': self.ep,
            'eo': self.eo,
        })

    def inv_cubie_cube(self):
        """
        void invCubieCube() {
            for (byte edge=0; edge<12; edge++)
                temps.ep[ep[edge]] = edge;
            for (byte edge=0; edge<12; edge++)
                temps.eo[edge] = eo[temps.ep[edge]];
            for (byte corn=0; corn<8; corn++)
                temps.cp[cp[corn]] = corn;
            for (byte corn=0; corn<8; corn++) {
                byte ori = co[temps.cp[corn]];
                temps.co[corn] = (byte) -ori;
                if (temps.co[corn] < 0)
                    temps.co[corn] += 3;
            }
            copy(temps);
        }
        """
        temps = CubieCube()

        for i in range(12):
            temps.ep[self.ep[i]] = i
        for i in range(12):
            temps.eo[i] = self.eo[temps.ep[i]]
        for i in range(8):
            temps.cp[self.cp[i]] = i
        for i in range(8):
            temps.co[i] = -self.co[temps.cp[i]]  # note negative
            if temps.co[i] < 0:
                temps.co[i] += 3

        self.copy(temps)

    def urf_conjugate(self):
        """
        this = S_urf^-1 * this * S_urf.

        void URFConjugate() {
            if (temps == null) {
                temps = new CubieCube();
            }
            CornMult(urf2, this, temps);
            CornMult(temps, urf1, this);
            EdgeMult(urf2, this, temps);
            EdgeMult(temps, urf1, this);
        }
        """
        temps = CubieCube()
        corner_multiply(URF2, self, temps)
        corner_multiply(temps, URF1, self)
        edge_multiply(URF2, self, temps)
        edge_multiply(temps, URF1, self)

    """
    // ************************ Get and set coordinates ************************
    // XSym : Symmetry Coordinate of X. MUST be called after
        initialization of ClassIndexToRepresentantArrays.

    // ++++++++++++++++++++ Phase 1 Coordinates ++++++++++++++++++++
    // Flip : Orientation of 12 Edges. Raw[0, 2048) Sym[0, 336 * 8)
    // Twist : Orientation of 8 Corners. Raw[0, 2187) Sym[0, 324 * 8)
    // UDSlice : Positions of the 4 UDSlice edges, the order is ignored. [0, 495)
    """

    def get_flip(self) -> int:
        """
        int getFlip() {
            int idx = 0;
            for (int i=0; i<11; i++) {
                idx <<= 1;
                idx |= eo[i];
            }
            return idx;
        }
        """
        idx = 0
        for i in range(11):
            idx <<= 1
            idx |= self.eo[i]
        return idx

    def set_flip(self, idx: int):
        """
        void setFlip(int idx) {
            int parity = 0;
            for (int i=10; i>=0; i--) {
                parity ^= eo[i] = (byte) (idx & 1);
                idx >>= 1;
            }
            eo[11] = (byte)parity;
        }
        """
        parity = 0
        self.eo = [None] * 12
        for i in range(10, -1, -1):
            self.eo[i] = idx & 1
            parity ^= self.eo[i]
            idx >>= 1
        self.eo[11] = parity

    def get_flip_sym(self):
        """
        int getFlipSym() {
            if (FlipR2S != null) {
                return FlipR2S[getFlip()];
            }
            if (temps == null) {
                temps = new CubieCube();
            }
            for (int k=0; k<16; k+=2) {
                EdgeConjugate(this, SymInv[k], temps);
                int idx = Util.binarySearch(FlipS2R, temps.getFlip());
                if (idx != 0xffff) {
                    return (idx << 3) | (k >> 1);
                }
            }
            return 0;
        }
        """
        if FLIP_R2S is not None:
            return FLIP_R2S[self.get_flip()]

        raise ValueError("not FLIP_R2S")
        temps = CubieCube()

        for k in range(16):
            edge_conjugate(self, SYM_INV[k], temps)
            flip = temps.get_flip()
            try:
                idx = FLIP_S2R.index(flip)
            except ValueError:
                continue
            else:
                return (idx << 3) | (k >> 1)
        return 0

    def get_twist(self) -> int:
        """
        int getTwist() {
            int idx = 0;
            for (int i=0; i<7; i++) {
                idx *= 3;
                idx += co[i];
            }
            return idx;
        }
        """
        idx = 0
        for i in range(7):
            idx *= 3
            idx += self.co[i]
        return idx

    def set_twist(self, idx: int):
        """
        void setTwist(int idx) {
            int twst = 0;
            for (int i=6; i>=0; i--) {
                twst += co[i] = (byte) (idx % 3);
                idx /= 3;
            }
            co[7] = (byte) ((15 - twst) % 3);
        }
        """
        twist = 0
        self.co = [None] * 8
        for i in range(6, -1, -1):
            self.co[i] = idx % 3
            twist += self.co[i]
            idx //= 3
        self.co[7] = ((15 - twist) % 3)

    def get_twist_sym(self) -> int:
        """
        int getTwistSym() {
            if (TwistR2S != null) {
                return TwistR2S[getTwist()];
            }
            if (temps == null) {
                temps = new CubieCube();
            }
            for (int k=0; k<16; k+=2) {
                CornConjugate(this, SymInv[k], temps);
                int idx = Util.binarySearch(TwistS2R, temps.getTwist());e
                if (idx != 0xffff) {
                    return (idx << 3) | (k >> 1);
                }
            }
            return 0;
        }
        """
        if TWIST_R2S is not None:
            return TWIST_R2S[self.get_twist()]

        temps = CubieCube()

        for k in range(0, 16, 2):
            corner_conjugate(self, SYM_INV[k], temps)
            twist = temps.get_twist()
            try:
                idx = TWIST_S2R.index(twist)
            except ValueError:
                continue
            else:
                return (idx << 3) | (k >> 1)
        return 0

    def get_ud_slice(self) -> int:
        # http://kociemba.org/math/UDSliceCoord.htm
        return util.get_comb(self.ep, 8)

    def set_ud_slice(self, idx: int):
        util.set_comb(self.ep, idx, 8)

    def get_u4_comb(self) -> int:
        return util.get_comb(self.ep, 0)

    def get_d4_comb(self) -> int:
        return util.get_comb(self.ep, 4)

    """
    // ++++++++++++++++++++ Phase 2 Coordinates ++++++++++++++++++++
    // EPerm : Permutations of 8 UD Edges. Raw[0, 40320) Sym[0, 2187 * 16)
    // Cperm : Permutations of 8 Corners. Raw[0, 40320) Sym[0, 2187 * 16)
    // MPerm : Permutations of 4 UDSlice Edges. [0, 24)
    """

    def get_c_perm(self) -> int:
        return util.get_8_perm(self.cp)

    def set_c_perm(self, idx: int):
        util.set_8_perm(self.cp, idx)

    def get_c_perm_sym(self) -> int:
        """
        int getCPermSym() {
            if (EPermR2S != null) {
                int idx = EPermR2S[getCPerm()];
                idx ^= e2c[idx&0x0f];
                return idx;
            }
            if (temps == null) {
                temps = new CubieCube();
            }
            for (int k=0; k<16; k++) {
                CornConjugate(this, SymInv[k], temps);
                int idx = Util.binarySearch(EPermS2R, temps.getCPerm());
                if (idx != 0xffff) {
                    return (idx << 4) | k;
                }
            }
            return 0;
        }
        """
        if E_PERM_R2S is not None:
            idx = E_PERM_R2S[self.get_c_perm()]
            idx ^= E2C[idx & 0x0f]
            return idx

        temps = CubieCube()

        for k in range(16):
            corner_conjugate(self, SYM_INV[k], temps)
            c_perm = temps.get_c_perm()
            try:
                idx = E_PERM_S2R.index(c_perm)
            except ValueError:
                continue
            else:
                return (idx << 4) | k
        return 0

    def get_e_perm(self) -> int:
        # return util.get_n_perm(self.ep, 12)
        return util.get_8_perm(self.ep)

    def set_e_perm(self, idx: int):
        # self.ep = util.set_n_perm(idx, 12)
        # self.ep = util.set_8_perm(idx) + [0, 0, 0, 0]  # using 8 perm????
        util.set_8_perm(self.ep, idx)  # using 8 perm????

    def get_e_perm_sym(self) -> int:
        """
        int getEPermSym() {
            if (EPermR2S != null) {
                return EPermR2S[getEPerm()];
            }
            if (temps == null) {
                temps = new CubieCube();
            }
            for (int k=0; k<16; k++) {
                EdgeConjugate(this, SymInv[k], temps);
                int idx = Util.binarySearch(EPermS2R, temps.getEPerm());
                if (idx != 0xffff) {
                    return (idx << 4) | k;
                }
            }
            return 0;
        }
        """
        if E_PERM_R2S is not None:
            return E_PERM_R2S[self.get_e_perm()]

        temps = CubieCube()

        for k in range(16):
            edge_conjugate(self, SYM_INV[k], temps)
            e_perm = temps.get_e_perm()
            try:
                idx = E_PERM_S2R.index(e_perm)
            except ValueError:
                continue
            else:
                return (idx << 4) | k
        return 0

    def get_m_perm(self) -> int:
        return util.get_comb(self.ep, 8) >> 9

    def set_m_perm(self, idx: int):
        util.set_comb(self.ep, idx << 9, 8)

    def verify(self):
        """
        Check a cubiecube for solvability.

        The java version returned the following codes. To be more pythonic,
        this raises a value error if the cube is not solvable and returns
        clean if it is solvable.

           0: Cube is solvable
          -2: Not all 12 edges exist exactly once
          -3: Flip error: One edge has to be flipped
          -4: Not all corners exist exactly once
          -5: Twist error: One corner has to be twisted
          -6: Parity error: Two corners or two edges have to be exchanged

        int verify() {
            int sum = 0;
            int edgeMask = 0;
            for (int e=0; e<12; e++)
                edgeMask |= (1 << ep[e]);
            if (edgeMask != 0x0fff)
                return -2;// missing edges
            for (int i=0; i<12; i++)
                sum ^= eo[i];
            if (sum % 2 != 0)
                return -3;
            int cornMask = 0;
            for (int c=0; c<8; c++)
                cornMask |= (1 << cp[c]);
            if (cornMask != 0x00ff)
                return -4;// missing corners
            sum = 0;
            for (int i=0; i<8; i++)
                sum += co[i];
            if (sum % 3 != 0)
                return -5;// twisted corner
            if ((Util.getNParity(Util.getNPerm(ep, 12), 12) ^ Util.getNParity(getCPerm(), 8)) != 0)
                return -6;// parity error
            return 0;// cube ok
        }
        """
        edge_mask = 0
        for i in range(12):
            edge_mask |= 1 << self.ep[i]

        if edge_mask != 0x0fff:
            raise ValueError("-2: Not all 12 edges exist exactly once")

        flip_sum = 0
        for i in range(12):
            flip_sum ^= self.eo[i]

        if flip_sum % 2 != 0:
            raise ValueError("-3: Flip error: One edge has to be flipped")

        corn_mask = 0
        for i in range(8):
            corn_mask |= 1 << self.cp[i]

        if corn_mask != 0x00ff:
            raise ValueError("-4: Not all corners exist exactly once")

        twist_sum = 0
        for i in range(8):
            twist_sum += self.co[i]

        if twist_sum % 3 != 0:
            raise ValueError("-5: Twist error: One corner has to be twisted")

        ep_parity = util.get_n_parity(util.get_n_perm(self.ep, 12), 12)
        c_perm_parity = util.get_n_parity(self.get_c_perm(), 8)
        if (ep_parity ^ c_perm_parity) != 0:
            raise ValueError(
                "-6: Parity error: Two corners or two edges have to be exchanged",
                util.get_n_perm(self.ep, 12),
                ep_parity,
                self.get_c_perm(),
                c_perm_parity,
            )


def corner_multiply(a: CubieCube, b: CubieCube, prod: CubieCube):
    """
    prod = a * b, Corner Only.

    static void CornMult(CubieCube a, CubieCube b, CubieCube prod) {
        for (int corn=0; corn<8; corn++) {
            prod.cp[corn] = a.cp[b.cp[corn]];
            byte oriA = a.co[b.cp[corn]];
            byte oriB = b.co[corn];
            byte ori = oriA;
            ori += (oriA<3) ? oriB : 6-oriB;
            ori %= 3;
            if ((oriA >= 3) ^ (oriB >= 3)) {
                ori += 3;
            }
            prod.co[corn] = ori;
        }
    }
    """
    for i in range(8):
        prod.cp[i] = a.cp[b.cp[i]]
        ori_a = a.co[b.cp[i]]
        ori_b = b.co[i]
        ori = ori_a
        ori += ori_b if (ori_a < 3) else 6 - ori_b
        ori %= 3
        if (ori_a >= 3) ^ (ori_b >= 3):
            ori += 3
        prod.co[i] = ori


def edge_multiply(a: CubieCube, b: CubieCube, prod: CubieCube):
    """
    prod = a * b, Edge Only.

    static void EdgeMult(CubieCube a, CubieCube b, CubieCube prod) {
        for (int ed=0; ed<12; ed++) {
            prod.ep[ed] = a.ep[b.ep[ed]];
            prod.eo[ed] = (byte) (b.eo[ed] ^ a.eo[b.ep[ed]]);
        }
    }
    """
    for i in range(12):
        prod.ep[i] = a.ep[b.ep[i]]
        prod.eo[i] = b.eo[i] ^ a.eo[b.ep[i]]


def corner_conjugate(a: CubieCube, idx: int, b: CubieCube):
    """
    b = S_idx^-1 * a * S_idx, Corner Only.

    static void CornConjugate(CubieCube a, int idx, CubieCube b) {
        CubieCube sinv = CubeSym[SymInv[idx]];
        CubieCube s = CubeSym[idx];
        for (int corn=0; corn<8; corn++) {
            b.cp[corn] = sinv.cp[a.cp[s.cp[corn]]];
            byte oriA = sinv.co[a.cp[s.cp[corn]]];
            byte oriB = a.co[s.cp[corn]];
            b.co[corn] = (byte) ((oriA<3) ? oriB : (3-oriB) % 3);
        }
    }
    """
    s_inv = CUBE_SYM[SYM_INV[idx]]
    s = CUBE_SYM[idx]
    for i in range(8):
        b.cp[i] = s_inv.cp[a.cp[s.cp[i]]]
        ori_a = s_inv.co[a.cp[s.cp[i]]]
        ori_b = a.co[s.cp[i]]
        b.co[i] = ori_b if ori_a < 3 else (3 - ori_b) % 3


def edge_conjugate(a: CubieCube, idx: int, b: CubieCube):
    """
    b = S_idx^-1 * a * S_idx, Edge Only.

    static void EdgeConjugate(CubieCube a, int idx, CubieCube b) {
        CubieCube sinv = CubeSym[SymInv[idx]];
        CubieCube s = CubeSym[idx];
        for (int ed=0; ed<12; ed++) {
            b.ep[ed] = sinv.ep[a.ep[s.ep[ed]]];
            b.eo[ed] = (byte) (s.eo[ed] ^ a.eo[s.ep[ed]] ^ sinv.eo[a.ep[s.ep[ed]]]);
        }
    }
    """
    s_inv = CUBE_SYM[SYM_INV[idx]]
    s = CUBE_SYM[idx]
    for ed in range(12):
        b.ep[ed] = s_inv.ep[a.ep[s.ep[ed]]]
        b.eo[ed] = (s.eo[ed] ^ a.eo[s.ep[ed]] ^ s_inv.eo[a.ep[s.ep[ed]]])


# **************** Initialization functions *********************

CUBE_SYM: List[CubieCube] = [None] * 16  # 16 symmetries generated by S_F2, S_U4 and S_LR2
MOVE_CUBE: List[CubieCube] = [None] * 18  # 18 move cubes

SYM_INV = None
SYM_MULT = None
SYM_MOVE = None
SYM_8_MULT = None
SYM_8_MOVE = None
SYM_8_MULT_INV = None
SYM_MOVE_UD = None

URF1 = CubieCube(2531, 1373, 67026819, 1367)
URF2 = CubieCube(2089, 1906, 322752913, 2040)

URF_MOVE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    [6, 7, 8, 0, 1, 2, 3, 4, 5, 15, 16, 17, 9, 10, 11, 12, 13, 14],
    [3, 4, 5, 6, 7, 8, 0, 1, 2, 12, 13, 14, 15, 16, 17, 9, 10, 11],
    [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17, 16, 15],
    [8, 7, 6, 2, 1, 0, 5, 4, 3, 17, 16, 15, 11, 10, 9, 14, 13, 12],
    [5, 4, 3, 8, 7, 6, 2, 1, 0, 14, 13, 12, 17, 16, 15, 11, 10, 9],
]

URF_MOVE_INV = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    [3, 4, 5, 6, 7, 8, 0, 1, 2, 12, 13, 14, 15, 16, 17, 9, 10, 11],
    [6, 7, 8, 0, 1, 2, 3, 4, 5, 15, 16, 17, 9, 10, 11, 12, 13, 14],
    [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17, 16, 15],
    [5, 4, 3, 8, 7, 6, 2, 1, 0, 14, 13, 12, 17, 16, 15, 11, 10, 9],
    [8, 7, 6, 2, 1, 0, 5, 4, 3, 17, 16, 15, 11, 10, 9, 14, 13, 12],
]


def init_move():
    """
    static void initMove() {
        moveCube[0] = new CubieCube(15120, 0, 119750400, 0);
        moveCube[3] = new CubieCube(21021, 1494, 323403417, 0);
        moveCube[6] = new CubieCube(8064, 1236, 29441808, 550);
        moveCube[9] = new CubieCube(9, 0, 5880, 0);
        moveCube[12] = new CubieCube(1230, 412, 2949660, 0);
        moveCube[15] = new CubieCube(224, 137, 328552, 137);
        for (int a=0; a<18; a+=3) {
            for (int p=0; p<2; p++) {
                moveCube[a+p+1] = new CubieCube();
                EdgeMult(moveCube[a+p], moveCube[a], moveCube[a+p+1]);
                CornMult(moveCube[a+p], moveCube[a], moveCube[a+p+1]);
            }
        }
    }
    """
    global MOVE_CUBE

    MOVE_CUBE[0] = CubieCube(15120, 0, 119750400, 0)
    MOVE_CUBE[3] = CubieCube(21021, 1494, 323403417, 0)
    MOVE_CUBE[6] = CubieCube(8064, 1236, 29441808, 550)
    MOVE_CUBE[9] = CubieCube(9, 0, 5880, 0)
    MOVE_CUBE[12] = CubieCube(1230, 412, 2949660, 0)
    MOVE_CUBE[15] = CubieCube(224, 137, 328552, 137)

    for a, p in itertools.product(range(0, 18, 3), range(2)):
        MOVE_CUBE[a + p + 1] = CubieCube()
        edge_multiply(MOVE_CUBE[a + p], MOVE_CUBE[a], MOVE_CUBE[a + p + 1])
        corner_multiply(MOVE_CUBE[a + p], MOVE_CUBE[a], MOVE_CUBE[a + p + 1])


def init_sym():
    global SYM_INV, SYM_MULT, SYM_MOVE, SYM_8_MULT, SYM_8_MOVE, SYM_8_MULT_INV, SYM_MOVE_UD

    SYM_INV = [0] * 16
    SYM_MULT = [[0] * 16 for _ in range(16)]
    SYM_MOVE = [[0] * 18 for _ in range(16)]
    SYM_8_MULT = [[0] * 8 for _ in range(8)]
    SYM_8_MOVE = [[0] * 18 for _ in range(8)]
    SYM_8_MULT_INV = [[0] * 8 for _ in range(8)]
    SYM_MOVE_UD = [[0] * 10 for _ in range(16)]

    c = CubieCube()
    d = CubieCube()

    f2 = CubieCube(28783, 0, 259268407, 0)
    u4 = CubieCube(15138, 0, 119765538, 7)
    lr2 = CubieCube(5167, 0, 83473207, 0)
    lr2.co = [3, 3, 3, 3, 3, 3, 3, 3]

    for i in range(16):
        CUBE_SYM[i] = CubieCube(c)
        corner_multiply(c, u4, d)
        edge_multiply(c, u4, d)
        c, d = d, c

        if i % 4 == 3:
            corner_multiply(c, lr2, d)
            edge_multiply(c, lr2, d)
            c, d = d, c

        if i % 8 == 7:
            corner_multiply(c, f2, d)
            edge_multiply(c, f2, d)
            c, d = d, c

    for i, j in itertools.product(range(16), range(16)):
        corner_multiply(CUBE_SYM[i], CUBE_SYM[j], c)
        for k in range(16):
            if (CUBE_SYM[k].cp[0] == c.cp[0] and
                    CUBE_SYM[k].cp[1] == c.cp[1] and CUBE_SYM[k].cp[2] == c.cp[2]):
                SYM_MULT[i][j] = k
                if k == 0:
                    SYM_INV[i] = j
                break

    for j, s in itertools.product(range(18), range(16)):
        corner_conjugate(MOVE_CUBE[j], SYM_INV[s], c)

        for m in range(18):
            if c.cp[0:8:2] != MOVE_CUBE[m].cp[0:8:2]:
                continue
            SYM_MOVE[s][j] = m
            break

    for j, s in itertools.product(range(10), range(16)):
        SYM_MOVE_UD[s][j] = util.STD_2_UD[SYM_MOVE[s][util.UD_2_STD[j]]]

    for j, s in itertools.product(range(8), range(8)):
        SYM_8_MULT[j][s] = SYM_MULT[j << 1][s << 1] >> 1
        SYM_8_MULT_INV[j][s] = SYM_MULT[j << 1][SYM_INV[s << 1]] >> 1

    for j, s in itertools.product(range(18), range(8)):
        SYM_8_MOVE[s][j] = SYM_MOVE[s << 1][j]


def init_flip_sym_2_raw():
    """
    static void initFlipSym2Raw() {
        CubieCube c = new CubieCube();
        CubieCube d = new CubieCube();
        int[] occ = new int[2048 >> 5];
        int count = 0;
        for (int i=0; i<2048>>5; occ[i++] = 0);
        FlipR2S = new char[2048];
        for (int i=0; i<2048; i++) {
            if ((occ[i>>5]&(1<<(i&0x1f))) == 0) {
                c.setFlip(i);
                for (int s=0; s<16; s+=2) {
                    EdgeConjugate(c, s, d);
                    int idx = d.getFlip();
                    if (idx == i) {
                        SymStateFlip[count] |= 1 << (s >> 1);
                    }
                    occ[idx>>5] |= 1<<(idx&0x1f);
                    FlipR2S[idx] = (char) ((count << 3) | (s >> 1));
                }
                FlipS2R[count++] = (char) i;
            }
        }
    }
    """
    global SYM_STATE_FLIP, FLIP_R2S, FLIP_S2R

    c = CubieCube()
    d = CubieCube()
    count = 0

    occ = [0] * 64  # array.ArrayType('L', (0 for _ in range(64)))
    SYM_STATE_FLIP = [0] * 336  # array.ArrayType('h', (0 for _ in range(336)))
    FLIP_R2S = [None] * 2048  # array.ArrayType('h', (0 for _ in range(2048)))
    FLIP_S2R = [None] * 336  # array.ArrayType('h', (0 for _ in range(336)))

    for i in range(2048):
        if (occ[i >> 5] & (1 << (i & 0x1f))) == 0:
            c.set_flip(i)
            for s in range(0, 16, 2):
                edge_conjugate(c, s, d)
                idx = d.get_flip()
                if idx == i:
                    SYM_STATE_FLIP[count] |= 1 << (s >> 1)
                occ[idx >> 5] |= 1 << (idx & 0x1f)
                FLIP_R2S[idx] = (count << 3) | (s >> 1)
            FLIP_S2R[count] = i
            count += 1


def init_twist_sym_2_raw():
    """
    static void initTwistSym2Raw() {
        CubieCube c = new CubieCube();
        CubieCube d = new CubieCube();
        int[] occ = new int[2187/32+1];
        int count = 0;
        for (int i=0; i<2187/32+1; occ[i++] = 0);
        TwistR2S = new char[2187];
        for (int i=0; i<2187; i++) {
            if ((occ[i>>5]&(1<<(i&0x1f))) == 0) {
                c.setTwist(i);
                for (int s=0; s<16; s+=2) {
                    CornConjugate(c, s, d);
                    int idx = d.getTwist();
                    if (idx == i) {
                        SymStateTwist[count] |= 1 << (s >> 1);
                    }
                    occ[idx>>5] |= 1<<(idx&0x1f);
                    TwistR2S[idx] = (char) ((count << 3) | (s >> 1));
                }
                TwistS2R[count++] = (char) i;
            }
        }
    }
    """
    global SYM_STATE_TWIST, TWIST_R2S, TWIST_S2R

    c = CubieCube()
    d = CubieCube()
    count = 0

    occ = array.ArrayType('L', (0 for _ in range(69)))
    SYM_STATE_TWIST = array.ArrayType('h', (0 for _ in range(324)))
    TWIST_R2S = array.ArrayType('h', (0 for _ in range(2187)))
    TWIST_S2R = array.ArrayType('h', (0 for _ in range(324)))

    for i in range(2187):
        if (occ[i >> 5] & (1 << (i & 0x1f))) == 0:
            c.set_twist(i)
            for s in range(0, 16, 2):
                corner_conjugate(c, s, d)
                idx = d.get_twist()
                if idx == i:
                    SYM_STATE_TWIST[count] |= 1 << (s >> 1)
                occ[idx >> 5] |= 1 << (idx & 0x1f)
                TWIST_R2S[idx] = (count << 3) | (s >> 1)
            TWIST_S2R[count] = i
            count += 1


def init_perm_sym_2_raw():
    """
    static void initPermSym2Raw() {
        CubieCube c = new CubieCube();
        CubieCube d = new CubieCube();
        int[] occ = new int[40320 / 32];
        int count = 0;
        for (int i=0; i<40320/32; occ[i++] = 0);
        EPermR2S = new char[40320];
        for (int i=0; i<40320; i++) {
            if ((occ[i>>5]&(1<<(i&0x1f))) == 0) {
                c.setEPerm(i);
                for (int s=0; s<16; s++) {
                    EdgeConjugate(c, s, d);
                    int idx = d.getEPerm();
                    if (idx == i) {
                        SymStatePerm[count] |= 1 << s;
                    }
                    occ[idx>>5] |= 1<<(idx&0x1f);
                    int a = d.getU4Comb();
                    int b = d.getD4Comb() >> 9;
                    int m = 494 - (a & 0x1ff) + (a >> 9) * 70 + b * 1680;
                    MtoEPerm[m] = EPermR2S[idx] = (char) (count << 4 | s);
                }
                EPermS2R[count++] = (char) i;
            }
        }
    }
    """
    global E_PERM_R2S, E_PERM_S2R, M2E_PERM, SYM_STATE_PERM

    c = CubieCube()
    d = CubieCube()
    count = 0

    occ = array.ArrayType('L', (0 for _ in range(40320 // 32)))
    E_PERM_R2S = array.ArrayType('H', (0 for _ in range(40320)))
    E_PERM_S2R = array.ArrayType('H', (0 for _ in range(2768)))
    M2E_PERM = array.ArrayType('H', (0 for _ in range(40320)))
    SYM_STATE_PERM = array.ArrayType('H', (0 for _ in range(2768)))

    for i in range(40320):
        if (occ[i >> 5] & (1 << (i & 0x1f))) == 0:
            c.set_e_perm(i)
            for s in range(16):
                edge_conjugate(c, s, d)
                idx = d.get_e_perm()
                if idx == i:
                    SYM_STATE_PERM[count] |= 1 << s
                occ[idx >> 5] |= 1 << (idx & 0x1f)
                a = d.get_u4_comb()
                b = d.get_d4_comb() >> 9
                m = 494 - (a & 0x1ff) + (a >> 9) * 70 + b * 1680
                M2E_PERM[m] = E_PERM_R2S[idx] = (count << 4 | s)
            E_PERM_S2R[count] = i
            count += 1
