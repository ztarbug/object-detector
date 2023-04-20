# How-To Create ffmpeg command

- Get .csv (Strings cannot be escaped!)
- Replace `(\d+),(\S+),(\S+)` with `ffmpeg -nostdin -rtsp_transport tcp -i '$3' -c:v copy -an $2.mp4 >$2.log 2>&1 & \`
  or run
  `sed -E 's/([^[:space:]]+);([^[:space:]]+)/ffmpeg -nostdin -rtsp_transport tcp -i "\2" -c:v copy -an -t 00:30:00 -y \1.mp4 >\1.log 2>\&1 \& \\/' Capture_Cams.csv > record_script.sh`

# How-To transmit data
- Encrypt symmetrically
  - Set if there are errors
    `GPG_TTY=$(tty);export GPG_TTY`
  - Encrypt tar in-place
    `tar cvf - recordings/ | gpg -c -o files.tar.gpg --compress-algo none`
- Send with (using BBR in case of packet-loss-ridden networks)
  `cat file.tar.gpg | socat - tcp4:<ip>:<port1>,setsockopt-string=6:13:bbr`
- If relay needed (b/c of NAT or Firewall)
  - Forward with
    `socat tcp4-listen:<port2>,fork,reuseaddr tcp4-listen:<port1>`
  - Receive with
    `socat - tcp4:<ip>:<port2> | pv > file.tar.gpg`
- else
  - Receive with
    `socat - tcp4-listen:<port1>,fork,reuseaddr | pv > file.tar.gpg`

- Decrypt
  `gpg -d file.tar.gpg > file.tar`