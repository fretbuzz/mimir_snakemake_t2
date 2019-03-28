rm /testing.txt
cp /etc/resolv.conf /testing.txt
sed  -i s/127.0.0.11/8.8.8.8/ /testing.txt
cat /testing.txt > /etc/resolv.conf
cat /etc/resolv.conf