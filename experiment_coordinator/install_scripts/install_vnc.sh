sudo apt-get update
sudo apt install -y xfce4 xfce4-goodies tightvncserver
# this next command is giving me trouble. So still a work in progress
{ echo 'hellothe\n';
  echo 'hellothe\n';
  echo 'n\n';
} | vncserver
vncserver -kill :1
mv ~/.vnc/xstartup ~/.vnc/xstartup.bak

touch ~/.vnc/xstartup
echo "#!/bin/bash
xrdb $HOME/.Xresources
startxfce4 &" >>  ~/.vnc/xstartup

sudo chmod +x ~/.vnc/xstartup
vncserver
