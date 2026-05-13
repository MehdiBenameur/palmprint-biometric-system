using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Konscious.Security.Cryptography;
using Palmprint.Application.Interfaces;

namespace Palmprint.Infrastructure.Security;

public class TemplateSecurityService : ITemplateSecurityService
{
    private readonly byte[] _encryptionKey;

    public TemplateSecurityService()
    {
        var key = Environment.GetEnvironmentVariable("PALMPRINT_AES_KEY");

        if (string.IsNullOrWhiteSpace(key))
        {
            _encryptionKey = SHA256.HashData(Encoding.UTF8.GetBytes("development-key-change-this"));
        }
        else
        {
            _encryptionKey = Convert.FromBase64String(key);
        }

        if (_encryptionKey.Length != 32)
            throw new InvalidOperationException("AES key must be 32 bytes for AES-256.");
    }

    public byte[] EncryptEmbedding(float[] embedding)
    {
        var json = JsonSerializer.Serialize(embedding);
        var plainBytes = Encoding.UTF8.GetBytes(json);

        using var aes = Aes.Create();
        aes.Key = _encryptionKey;
        aes.GenerateIV();
        aes.Mode = CipherMode.CBC;
        aes.Padding = PaddingMode.PKCS7;

        using var encryptor = aes.CreateEncryptor();
        var cipherBytes = encryptor.TransformFinalBlock(plainBytes, 0, plainBytes.Length);

        var result = new byte[aes.IV.Length + cipherBytes.Length];

        Buffer.BlockCopy(aes.IV, 0, result, 0, aes.IV.Length);
        Buffer.BlockCopy(cipherBytes, 0, result, aes.IV.Length, cipherBytes.Length);

        return result;
    }

    public string HashEmbedding(float[] embedding)
    {
        var json = JsonSerializer.Serialize(embedding);
        var bytes = Encoding.UTF8.GetBytes(json);

        using var argon2 = new Argon2id(bytes)
        {
            Salt = SHA256.HashData(Encoding.UTF8.GetBytes("palmprint-template-salt")),
            DegreeOfParallelism = 2,
            Iterations = 3,
            MemorySize = 65536
        };

        var hash = argon2.GetBytes(32);

        return Convert.ToBase64String(hash);
    }
}