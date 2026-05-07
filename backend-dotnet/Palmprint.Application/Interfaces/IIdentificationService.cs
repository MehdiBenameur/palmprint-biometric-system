using Palmprint.Application.DTOs;

namespace Palmprint.Application.Interfaces;

public interface IIdentificationService
{
    Task<IdentifyResponse> IdentifyAsync(IdentifyRequest request);
}