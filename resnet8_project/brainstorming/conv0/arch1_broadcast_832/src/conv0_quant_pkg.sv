// Generated: fixed pretrained Conv0 quantization, signed INT32 -> ReLU INT8.
package conv0_quant_pkg;
function automatic integer quant_m(input integer channel);
case (channel)
0: quant_m = 1145439378;
1: quant_m = 1146875512;
2: quant_m = 1499025764;
3: quant_m = 1688704632;
4: quant_m = 1689973370;
5: quant_m = 1282199442;
6: quant_m = 1171885099;
7: quant_m = 1433735231;
8: quant_m = 1924085427;
9: quant_m = 2051744702;
10: quant_m = 1439771906;
11: quant_m = 1744529516;
12: quant_m = 1558615247;
13: quant_m = 1273347095;
14: quant_m = 1256690059;
15: quant_m = 1242953877;
16: quant_m = 1273923047;
17: quant_m = 1705339820;
18: quant_m = 2115618556;
19: quant_m = 1093049293;
20: quant_m = 1094634201;
21: quant_m = 1423626006;
22: quant_m = 1905005356;
23: quant_m = 1165126611;
24: quant_m = 1783416464;
25: quant_m = 1545673184;
default: quant_m = 0;
endcase
endfunction
function automatic integer quant_s(input integer channel);
case (channel)
0: quant_s = 39;
1: quant_s = 40;
2: quant_s = 41;
3: quant_s = 42;
4: quant_s = 39;
5: quant_s = 38;
6: quant_s = 40;
7: quant_s = 39;
8: quant_s = 41;
9: quant_s = 42;
10: quant_s = 39;
11: quant_s = 39;
12: quant_s = 40;
13: quant_s = 38;
14: quant_s = 41;
15: quant_s = 40;
16: quant_s = 39;
17: quant_s = 40;
18: quant_s = 41;
19: quant_s = 38;
20: quant_s = 41;
21: quant_s = 40;
22: quant_s = 41;
23: quant_s = 40;
24: quant_s = 42;
25: quant_s = 41;
default: quant_s = 0;
endcase
endfunction
function automatic logic [7:0] quant_byte(input integer addr);
integer word_value;
begin
word_value = ((addr/4)%2 == 0) ? quant_m(addr/8) : quant_s(addr/8);
quant_byte = (word_value >> (8*(addr%4))) & 255;
end
endfunction
endpackage
