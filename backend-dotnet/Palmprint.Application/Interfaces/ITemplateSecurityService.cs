namespace Palmprint.Application.Interfaces;

public interface ITemplateSecurityService
{
    byte[] EncryptEmbedding(float[] embedding);
    string HashEmbedding(float[] embedding);
}