import unittest
from tomotwin.modules.training.argparse_ui import TrainingArgParseUI
class MyTestCase(unittest.TestCase):

    def test_parse_args_only_required(self):
        pdbpth = '/my/pdb/path/'
        volpth = '/my/vol/path/'
        outpth = '/my/output/path/'
        nc = '/path/to/config.json'
        ui = TrainingArgParseUI()
        parser = ui.create_parser()
        args = parser.parse_args(['-p', pdbpth, '-v', volpth, '-o', outpth, '-nc', nc])

        self.assertEqual(args.volpath, volpth)
        self.assertEqual(args.pdbpath, pdbpth)

    def test_parse_args_only_pdb_missing(self):
        pdbpth = '/my/pdb/path/'
        ui = TrainingArgParseUI()
        parser = ui.create_parser()
        failed = False
        try:
            parser.parse_args(['-p', pdbpth])
        except:
            failed = True
        self.assertEqual(True,failed, "Missing argument -v. It should fail.")

    def test_parse_args_only_vol_missing(self):
        volpth = '/my/vol/path/'
        ui = TrainingArgParseUI()
        parser = ui.create_parser()
        failed = False
        try:
            parser.parse_args(['-v', volpth])
        except:
            failed = True
        self.assertEqual(True,failed, "Missing argument -p. It should fail.")

    def test_parse_args_all(self):
        pdbpth = '/my/pdb/path/'
        volpth = '/my/vol/path/'
        outpth = '/my/output/path/'
        epochs = '10'
        nc = '/path/to/config.json'
        max_neg = '3'
        ui = TrainingArgParseUI()
        parser = ui.create_parser()
        args = parser.parse_args(['-p', pdbpth,
                                  '-v', volpth,
                                  '-o', outpth,
                                  '--epochs', epochs,
                                  '--max_neg', max_neg,
                                  '-nc', nc])

        self.assertEqual(args.volpath, volpth)
        self.assertEqual(args.pdbpath, pdbpth)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.max_neg, 3)
        self.assertEqual(args.outpath, outpth)

    def test_parse_args_all_no_max_neg(self):
        pdbpth = '/my/pdb/path/'
        volpth = '/my/vol/path/'
        outpth = '/my/output/path/'
        epochs = '10'
        nc = '/path/to/config.json'

        ui = TrainingArgParseUI()
        parser = ui.create_parser()
        args = parser.parse_args(['-p', pdbpth,
                                  '-v', volpth,
                                  '-o', outpth,
                                  '--epochs', epochs,
                                  '-nc', nc])

        self.assertEqual(args.volpath, volpth)
        self.assertEqual(args.pdbpath, pdbpth)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.max_neg, 1)

    def test_parse_args_all_run(self):
        pdbpth = '/my/pdb/path/'
        volpth = '/my/vol/path/'
        outpth = '/my/output/path/'
        nc = '/path/to/config.json'
        epochs = '10'
        ui = TrainingArgParseUI()
        args = ['-p', pdbpth,
                                  '-v', volpth,
                '-o', outpth,
                                  '--epochs', epochs,
                                  '-nc', nc]
        ui.run(args)
        tconf = ui.get_training_configuration()
        self.assertEqual(tconf.volume_path, volpth)
        self.assertEqual(tconf.pdb_path, pdbpth)
        self.assertEqual(tconf.num_epochs, 10)



