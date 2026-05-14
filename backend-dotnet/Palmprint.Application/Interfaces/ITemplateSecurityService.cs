namespace Palmprint.Application.Interfaces;

public interface ITemplateSecurityService
{
    byte[] EncryptEmbedding(float[] embedding);
    float[] DecryptEmbedding(byte[] encryptedEmbedding);
    string HashEmbedding(float[] embedding);
}