git clone https://github.com/gameontext/gameon.git
cd ./gameon
./go-admin.sh choose
eval $(./go-admin.sh env)
#ls
alias go-run
#ls
bash ./go-admin.sh setup
bash ./go-admin.sh up
./docker/go-run.sh wait
