/*
 * weight_crypto.c — AES-256-CTR encryption for model weights.
 *
 * Uses a minimal AES implementation (no OpenSSL dependency).
 * Key derivation: SHA-256(machine_id + salt) → AES-256 key.
 * Machine ID: combination of CPU info + hostname for hardware binding.
 *
 * Usage:
 *   Encrypt: weight_encrypt("weights.bin", "weights.enc", key)
 *   Decrypt: weight_decrypt_inplace(data, size, key)
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/weight_crypto.h"
#ifdef _WIN32
#include <windows.h>
#endif

/* ============ Minimal AES-256 (no dependency) ============ */

static const uint8_t sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static const uint8_t rcon[11] = {0,1,2,4,8,16,32,64,128,27,54};

static void aes256_key_expand(const uint8_t key[32], uint8_t rkeys[240]) {
    memcpy(rkeys, key, 32);
    uint8_t tmp[4];
    for (int i = 8; i < 60; i++) {
        memcpy(tmp, rkeys + (i-1)*4, 4);
        if (i % 8 == 0) {
            uint8_t t = tmp[0];
            tmp[0] = sbox[tmp[1]] ^ rcon[i/8];
            tmp[1] = sbox[tmp[2]];
            tmp[2] = sbox[tmp[3]];
            tmp[3] = sbox[t];
        } else if (i % 8 == 4) {
            for (int j = 0; j < 4; j++) tmp[j] = sbox[tmp[j]];
        }
        for (int j = 0; j < 4; j++)
            rkeys[i*4+j] = rkeys[(i-8)*4+j] ^ tmp[j];
    }
}

static void aes256_encrypt_block(const uint8_t rkeys[240], const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16];
    memcpy(s, in, 16);
    /* AddRoundKey 0 */
    for (int i = 0; i < 16; i++) s[i] ^= rkeys[i];
    /* Rounds 1-13 */
    for (int r = 1; r < 14; r++) {
        uint8_t t[16];
        /* SubBytes */
        for (int i = 0; i < 16; i++) t[i] = sbox[s[i]];
        /* ShiftRows */
        s[0]=t[0]; s[1]=t[5]; s[2]=t[10]; s[3]=t[15];
        s[4]=t[4]; s[5]=t[9]; s[6]=t[14]; s[7]=t[3];
        s[8]=t[8]; s[9]=t[13]; s[10]=t[2]; s[11]=t[7];
        s[12]=t[12]; s[13]=t[1]; s[14]=t[6]; s[15]=t[11];
        /* MixColumns (skip for last round) */
        if (r < 13) {
            for (int c = 0; c < 4; c++) {
                uint8_t a[4]; memcpy(a, s+c*4, 4);
                uint8_t h[4];
                for (int i = 0; i < 4; i++) h[i] = (a[i] & 0x80) ? (a[i]<<1)^0x1b : a[i]<<1;
                s[c*4+0] = h[0]^h[1]^a[1]^a[2]^a[3];
                s[c*4+1] = a[0]^h[1]^h[2]^a[2]^a[3];
                s[c*4+2] = a[0]^a[1]^h[2]^h[3]^a[3];
                s[c*4+3] = h[0]^a[0]^a[1]^a[2]^h[3];
            }
        }
        /* AddRoundKey */
        for (int i = 0; i < 16; i++) s[i] ^= rkeys[r*16+i];
    }
    /* Final round (no MixColumns) */
    uint8_t t[16];
    for (int i = 0; i < 16; i++) t[i] = sbox[s[i]];
    s[0]=t[0]; s[1]=t[5]; s[2]=t[10]; s[3]=t[15];
    s[4]=t[4]; s[5]=t[9]; s[6]=t[14]; s[7]=t[3];
    s[8]=t[8]; s[9]=t[13]; s[10]=t[2]; s[11]=t[7];
    s[12]=t[12]; s[13]=t[1]; s[14]=t[6]; s[15]=t[11];
    for (int i = 0; i < 16; i++) out[i] = s[i] ^ rkeys[14*16+i];
}

/* ============ AES-256-CTR mode ============ */

static void aes256_ctr(const uint8_t key[32], const uint8_t nonce[16],
                       uint8_t* data, size_t len) {
    uint8_t rkeys[240];
    aes256_key_expand(key, rkeys);

    uint8_t ctr[16], keystream[16];
    memcpy(ctr, nonce, 16);

    for (size_t i = 0; i < len; i += 16) {
        aes256_encrypt_block(rkeys, ctr, keystream);
        size_t block_len = (len - i < 16) ? len - i : 16;
        for (size_t j = 0; j < block_len; j++)
            data[i + j] ^= keystream[j];
        /* Increment counter */
        for (int k = 15; k >= 0; k--) {
            if (++ctr[k]) break;
        }
    }
}

/* ============ SHA-256 (minimal, for key derivation) ============ */

static const uint32_t sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define RR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z) (((x)&(y))^((~(x))&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x) (RR(x,2)^RR(x,13)^RR(x,22))
#define EP1(x) (RR(x,6)^RR(x,11)^RR(x,25))
#define SIG0(x) (RR(x,7)^RR(x,18)^((x)>>3))
#define SIG1(x) (RR(x,17)^RR(x,19)^((x)>>10))

void sha256(const uint8_t* data, size_t len, uint8_t hash[32]) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    /* Padding */
    size_t padded_len = ((len + 9 + 63) / 64) * 64;
    uint8_t* msg = (uint8_t*)calloc(padded_len, 1);
    memcpy(msg, data, len);
    msg[len] = 0x80;
    uint64_t bitlen = len * 8;
    for (int i = 0; i < 8; i++) msg[padded_len - 1 - i] = (uint8_t)(bitlen >> (i * 8));

    for (size_t chunk = 0; chunk < padded_len; chunk += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (msg[chunk+i*4]<<24)|(msg[chunk+i*4+1]<<16)|(msg[chunk+i*4+2]<<8)|msg[chunk+i*4+3];
        for (int i = 16; i < 64; i++)
            w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + EP1(e) + CH(e,f,g) + sha256_k[i] + w[i];
            uint32_t t2 = EP0(a) + MAJ(a,b,c);
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    free(msg);
    for (int i = 0; i < 8; i++) {
        hash[i*4] = h[i]>>24; hash[i*4+1] = h[i]>>16; hash[i*4+2] = h[i]>>8; hash[i*4+3] = h[i];
    }
}

/* ============ Hardware ID ============ */

static void get_machine_id(char* buf, int maxlen) {
    /* Combine hostname + CPU info for hardware binding */
    buf[0] = 0;
#ifdef _WIN32
    char hostname[256] = "unknown";
    DWORD sz = sizeof(hostname);
    GetComputerNameA(hostname, &sz);
    snprintf(buf, maxlen, "%s", hostname);
#else
    FILE* f;
    /* Hostname */
    f = fopen("/etc/hostname", "r");
    if (f) {
        if (fgets(buf, maxlen, f) == NULL) buf[0] = 0;
        fclose(f);
        if (buf[0]) buf[strcspn(buf, "\n")] = 0;
    }
    /* CPU ID from /proc/cpuinfo */
    f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "model name")) {
                strncat(buf, line, maxlen - strlen(buf) - 1);
                break;
            }
        }
        fclose(f);
    }
#endif
}

/* ============ Public API ============ */

/* Derive AES-256 key from license string + machine hardware */
void derive_key(const char* license, uint8_t key[32]) {
    char machine_id[512] = {0};
    get_machine_id(machine_id, sizeof(machine_id));

    /* key = SHA-256(license + "|" + machine_id) */
    size_t lic_len = strlen(license);
    size_t mid_len = strlen(machine_id);
    size_t total = lic_len + 1 + mid_len;
    uint8_t* combined = (uint8_t*)malloc(total);
    memcpy(combined, license, lic_len);
    combined[lic_len] = '|';
    memcpy(combined + lic_len + 1, machine_id, mid_len);
    sha256(combined, total, key);
    free(combined);
}

/* Encrypt weight file in-place (adds 16-byte nonce header) */
int weight_encrypt_file(const char* input, const char* output, const char* license) {
    uint8_t key[32];
    derive_key(license, key);

    FILE* fin = fopen(input, "rb");
    if (!fin) return -1;
    fseek(fin, 0, SEEK_END);
    size_t len = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(len);
    if (!data) {
        fclose(fin);
        return -1;
    }
    size_t read_len = fread(data, 1, len, fin);
    fclose(fin);
    if (read_len != len) {
        free(data);
        return -1;
    }

    /* Generate nonce from weight data hash (take first 16 bytes of SHA-256) */
    uint8_t nonce_hash[32];
    sha256(data, len > 1024 ? 1024 : len, nonce_hash);
    uint8_t nonce[16];
    memcpy(nonce, nonce_hash, 16);

    /* Encrypt */
    aes256_ctr(key, nonce, data, len);

    /* Write: nonce (16) + encrypted data */
    FILE* fout = fopen(output, "wb");
    if (!fout) { free(data); return -1; }
    fwrite("EFXE", 1, 4, fout);  /* magic: EFXE = EdgeFace eXtended Encrypted */
    fwrite(nonce, 1, 16, fout);
    fwrite(data, 1, len, fout);
    fclose(fout);
    free(data);
    return 0;
}

/* Decrypt weight data in-place (data starts after 20-byte header: 4 magic + 16 nonce) */
int weight_decrypt_inplace(uint8_t* raw, size_t raw_size, const char* license) {
    if (raw_size < 20 || memcmp(raw, "EFXE", 4) != 0) return -1;

    uint8_t key[32];
    derive_key(license, key);

    uint8_t* nonce = raw + 4;
    uint8_t* data = raw + 20;
    size_t data_len = raw_size - 20;

    aes256_ctr(key, nonce, data, data_len);

    /* Verify decryption by checking for EFXS magic in decrypted data */
    if (memcmp(data, "EFXS", 4) != 0) {
        /* Wrong key — re-encrypt to restore original state */
        aes256_ctr(key, nonce, data, data_len);
        return -2; /* invalid license */
    }

    return 0;
}

/* ============ CLI tool ============ */

#ifdef WEIGHT_CRYPTO_MAIN
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  Encrypt: %s encrypt input.bin output.enc LICENSE_KEY\n", argv[0]);
        fprintf(stderr, "  Test:    %s test encrypted.enc LICENSE_KEY\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "encrypt") == 0 && argc >= 5) {
        int r = weight_encrypt_file(argv[2], argv[3], argv[4]);
        if (r == 0) printf("Encrypted: %s -> %s\n", argv[2], argv[3]);
        else fprintf(stderr, "Error: %d\n", r);
        return r;
    }

    if (strcmp(argv[1], "test") == 0 && argc >= 4) {
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", argv[2]); return 1; }
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t* data = (uint8_t*)malloc(sz); fread(data, 1, sz, f); fclose(f);
        int r = weight_decrypt_inplace(data, sz, argv[3]);
        if (r == 0) printf("OK: decryption successful, EFXS magic verified\n");
        else if (r == -2) printf("FAIL: wrong license key for this machine\n");
        else printf("FAIL: not an encrypted file\n");
        free(data);
        return r == 0 ? 0 : 1;
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
}
#endif
