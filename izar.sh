# To use this script pass your EPFL user name and enter your password.
# Note: you need to enter the password twice (once for each folder).

rsync -v -r $1@izar.hpc.epfl.ch:/home/$1/ml-project-2-team-kangaroos/{saved,runs} .