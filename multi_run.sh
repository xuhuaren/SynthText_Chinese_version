for i in {1..10};do
python gen_para.py --id $i --viz &
done
echo "END"
