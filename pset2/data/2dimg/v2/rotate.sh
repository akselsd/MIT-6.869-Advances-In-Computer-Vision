for szFile in ./*.jpg
do 
    convert "$szFile" -rotate 180 "r/$(basename "$szFile")" ; 
done