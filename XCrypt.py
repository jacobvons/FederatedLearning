from GeneralFunc import stringify, destringify
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import PKCS1_v1_5
import math


def seg_encrypt(arr, pk, real_encrypt):
    """
    Segment encrypt a piece of data

    :param arr: some numpy array
    :param pk: public key used for encryption
    :param real_encrypt: really encrypt or not
    :return: encrypted bytes object of the original message
    """
    if real_encrypt:
        b_data = stringify(arr)
        cipher = PKCS1_v1_5.new(pk)
        block_size = math.ceil(pk.n.bit_length() / 8) - 11
        padded_msg = pad(b_data, block_size, style="pkcs7")
        encrypt_list = []

        for i in range(0, len(padded_msg), block_size):
            sub_msg = padded_msg[i: i + block_size]
            # sub = pk.encrypt(sub_msg, b"")
            sub = cipher.encrypt(sub_msg)
            encrypt_list.append(sub)
        text = b"".join(encrypt_list)
        return text
    else:
        return arr


def seg_decrypt(b_str_data, sk, real_encrypt, is_bias=False):
    """
    Segment decrypt some encrypted data

    :param is_bias: if it's bias. used for destringify
    :param b_str_data: some bytes object data to be decrypted
    :param sk: secret key used for decryption
    :param real_encrypt: real encrypt or not
    :return: decrypted text as a bytes object
    """
    if real_encrypt:
        encrypt_block = math.ceil(sk.n.bit_length() / 8)
        block_size = encrypt_block - 11
        cipher = PKCS1_v1_5.new(sk)
        decrypt_list = []

        for i in range(0, len(b_str_data), encrypt_block):
            sub_msg = b_str_data[i: i + encrypt_block]
            # sub = sk.decrypt(sub_msg)
            sub = cipher.decrypt(sub_msg, "DecryptError")
            decrypt_list.append(sub)

        padded = b"".join(decrypt_list)
        text = unpad(padded, block_size, style="pkcs7")
        arr = destringify(text, is_bias)
        return arr
    else:
        return b_str_data
