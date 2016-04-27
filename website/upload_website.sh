#!/usr/bin/env bash

echo "Compress all Html files"
tar -zcf website.tar.gz *

echo "Removing unused files"
sshpass -f ../password.txt ssh msd15090@i.cs.hku.hk "rm -rf public_html/*"

echo "Upload compress files"
sshpass -f ../password.txt scp website.tar.gz msd15090@i.cs.hku.hk:public_html/website.tar.gz

echo "Untar all files in the server"
sshpass -f ../password.txt ssh msd15090@i.cs.hku.hk "tar -zxf ~/public_html/website.tar.gz -C ~/public_html/"

echo "Set permission"
sshpass -f ../password.txt ssh msd15090@i.cs.hku.hk "chmod -R 777 ~/public_html"

echo "Remove temporary files from website"
sshpass -f ../password.txt ssh msd15090@i.cs.hku.hk "rm public_html/website.tar.gz"

echo "remove temporary files"
rm website.tar.gz