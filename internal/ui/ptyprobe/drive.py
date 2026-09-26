#!/usr/bin/env python3
"""Drive the line editor under a real pty and assert what lands on screen.

Terminal behaviour is the one part of the editor that unit tests cannot reach:
whether an escape sequence was fully consumed, whether the cursor ended up where
the arithmetic says it did. This runs ptyprobe under a pty, replays the byte
stream through a small terminal model, and checks the resulting line.

    go build -o /tmp/ptyprobe ./internal/ui/ptyprobe
    python3 internal/ui/ptyprobe/drive.py /tmp/ptyprobe
"""
import os, pty, select, sys, time

KEYS = {
 "CTRL_A":"\x01","CTRL_E":"\x05","CTRL_K":"\x0b","CTRL_U":"\x15","CTRL_W":"\x17",
 "CTRL_Y":"\x19","CTRL_T":"\x14","CTRL_R":"\x12","CTRL_P":"\x10","CTRL_N":"\x0e",
 "CTRL_LEFT":"\x1b[1;5D","CTRL_RIGHT":"\x1b[1;5C","ALT_B":"\x1bb","ALT_F":"\x1bf",
 "ALT_BS":"\x1b\x7f","ALT_D":"\x1bd","LEFT":"\x1b[D","RIGHT":"\x1b[C","UP":"\x1b[A",
 "HOME":"\x1b[H","END":"\x1b[F","DEL":"\x1b[3~","BS":"\x7f","CR":"\r",
 "PASTE_START":"\x1b[200~","PASTE_END":"\x1b[201~","ESC":"\x1b",
}

def run(script, cols=100):
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"]="xterm-256color"
        os.execv(sys.argv[1], [sys.argv[1]])
    import fcntl, termios, struct
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, cols, 0, 0))
    time.sleep(0.15)
    out=b""
    def drain(t=0.12):
        nonlocal out
        end=time.time()+t
        while time.time()<end:
            r,_,_=select.select([fd],[],[],0.02)
            if r:
                try: out+=os.read(fd,65536)
                except OSError: break
    drain(0.2)
    for item in script:
        os.write(fd, KEYS.get(item, item).encode())
        time.sleep(0.03)
        drain(0.05)
    drain(0.4)
    try: os.close(fd)
    except OSError: pass
    try: os.waitpid(pid,0)
    except ChildProcessError: pass
    return out.decode("utf-8","replace")

def screen(raw, cols=100):
    """Replay the byte stream through a tiny terminal model."""
    lines=[[" "]*cols]; row=0; col=0; i=0
    def put(ch):
        nonlocal col
        while len(lines[row])<=col: lines[row].append(" ")
        lines[row][col]=ch; col+=1
    while i<len(raw):
        c=raw[i]
        if c=="\x1b":
            j=i+1
            if j<len(raw) and raw[j]=="[":
                j+=1; p=""
                while j<len(raw) and not ("@"<=raw[j]<="~"): p+=raw[j]; j+=1
                f=raw[j] if j<len(raw) else ""
                n=int(p) if p.isdigit() else 1
                if f=="C": col+=n
                elif f=="D": col=max(0,col-n)
                elif f=="K":
                    for k in range(col,len(lines[row])): lines[row][k]=" "
                elif f=="J" and p=="2":
                    lines=[[" "]*cols]; row=0; col=0
                elif f=="H": row=0; col=0
                i=j+1; continue
            i=j+1; continue
        if c=="\r": col=0
        elif c=="\n":
            row+=1
            while len(lines)<=row: lines.append([" "]*cols)
        else: put(c)
        i+=1
    return [ "".join(l).rstrip() for l in lines ]

TESTS = {
 "ctrl-left does not leak ;5D": (["hello world foo","CTRL_LEFT","CTRL_LEFT","X","CR"], "hello Xworld foo"),
 "ctrl-w kills whole path":     (["cat internal/ui/editor.go","CTRL_W","done","CR"], "cat done"),
 "alt-backspace kills segment": (["cat internal/ui/editor.go","ALT_BS","md","CR"], "cat internal/ui/editor.md"),
 "ctrl-u then ctrl-y":          (["some text","CTRL_U","CTRL_Y","!","CR"], "some text!"),
 "ctrl-a ctrl-e":               (["world","CTRL_A","hello ","CTRL_E","!","CR"], "hello world!"),
 "alt-b alt-f word motion":     (["alpha beta gamma","ALT_B","ALT_B","X","CR"], "alpha Xbeta gamma"),
 "ctrl-t transpose":            (["teh","CTRL_T","CR"], "the"),
 "history up":                  (["UP","CR"], "git status"),
 "ctrl-r search":               (["CTRL_R","build","CR"], "go build ./..."),
 "bracketed paste multiline":   (["PASTE_START","one\ntwo","PASTE_END","CR"], "one\ntwo"),
 "paste strips colour codes":   (["PASTE_START","\x1b[31mred\x1b[0m ok","PASTE_END","CR"], "red ok"),
 "delete key":                  (["abc","CTRL_A","DEL","CR"], "bc"),
 "home end keys":               (["world","HOME","hi ","END","!","CR"], "hi world!"),
}

path=sys.argv[1]
fails=0
for name,(script,want) in TESTS.items():
    raw=run(script)
    got=None
    for line in screen(raw):
        if line.startswith("GOT["):
            got=line[4:-1] if line.endswith("]") else line[4:]
    if want.replace("\n","\n")==got or (("\n" in want) and got is not None and want.split("\n")[0] in got):
        print(f"  ok    {name}")
    else:
        fails+=1
        print(f"  FAIL  {name}\n        want {want!r}\n        got  {got!r}")
print("FAILURES:",fails)
