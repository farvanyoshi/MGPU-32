; bad_barrier.s - a deliberately broken kernel: warps reach different BAR instructions.
;
; args:   none
; launch: BLOCK_DIM >= 64 fails with STATUS 4 (barrier mismatch, ISA [BAR-2]);
;         a single-warp block passes, because one warp is always "all warps".
;
; Even warps wait at one barrier and odd warps at another. Under [BAR-3] this cannot
; deadlock; the second warp to arrive at a different PC raises the error instead.

        .kernel bad_barrier
bad_barrier:
        S_GETSR   s1, WARP_ID
        S_ANDI    s2, s1, 1
        S_BNZ     s2, odd
even:   BAR                         ; even warps wait here ...
        EXIT
odd:    BAR                         ; ... odd warps here: mismatch
        EXIT
