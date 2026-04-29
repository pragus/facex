#ifndef WEIGHT_CRYPTO_H
#define WEIGHT_CRYPTO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Encrypt weight file: input (EFXS) → output (EFXE + nonce + encrypted data) */
int weight_encrypt_file(const char* input, const char* output, const char* license);

/* Decrypt in-place. raw must start with EFXE magic.
 * On success: data at raw+20 contains decrypted EFXS payload.
 * Returns: 0 = OK, -1 = not encrypted, -2 = wrong key */
int weight_decrypt_inplace(uint8_t* raw, size_t raw_size, const char* license);

#ifdef __cplusplus
}
#endif

#endif
