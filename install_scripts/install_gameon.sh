git clone https://github.com/gameontext/gameon.git
cd ./gameon
{ echo '1\n';
} | ./go-admin.sh choose
eval $(./go-admin.sh env)
alias go-run
go-admin setup
go-admin up
