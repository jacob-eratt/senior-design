import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location('uart',Path(__file__).resolve().parents[1]/'rtl/kv260_conv0/tools/summarize_uart.py')
UART=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UART)

GOOD='\n'.join(['Conv0 KV260 benchmark: PL_HZ=100000000',
    'kind,mode,index,total_ns,run_cycles,raw_cycles,first_output_cycle,input_stalls,output_stalls,load_cycles',
    'FRAME,cold,0,30000,1300,1250,40,0,49,256',
    'FRAME,warm,0,25000,1300,1250,40,0,49,192',
    'PASS: 2 frames, 26624 INT8 comparisons per frame'])

class CaptureTests(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(len(UART.parse_capture(GOOD,1,1)),2)

    def test_rejections(self):
        cases=[GOOD.rsplit('\n',1)[0],GOOD+'\nFAIL,DMA wait',GOOD+'\nMismatch,index=0',
            GOOD+'\n'+GOOD,GOOD.replace('FRAME,warm,0','FRAME,warm,1'),
            GOOD.replace('FRAME,cold,0','FRAME,warm,0'),GOOD.replace('1300,1250','1200,1250'),
            GOOD.replace(',256',',257'),GOOD.replace(',30000,',',-1,'),
            GOOD.replace('2 frames','220 frames'),
            GOOD.replace('FRAME,warm,0,25000,1300,1250,40,0,49,192\n',''),
            GOOD+'\nFRAME,warm,1,25000,1300,1250,40,0,49,192']
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError): UART.parse_capture(value,1,1)

if __name__=='__main__': unittest.main()
