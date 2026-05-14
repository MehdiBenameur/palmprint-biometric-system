using Microsoft.AspNetCore.Mvc;
using Palmprint.Api.Models;
using Palmprint.Application.DTOs;
using Palmprint.Application.Interfaces;

namespace Palmprint.Api.Controllers;

[ApiController]
[Route("api/identification")]
public class IdentificationController : ControllerBase
{
    private readonly IIdentificationService _identificationService;

    public IdentificationController(IIdentificationService identificationService)
    {
        _identificationService = identificationService;
    }

    [HttpPost("identify")]
    [Consumes("multipart/form-data")]
    public async Task<IActionResult> Identify([FromForm] IdentifyFormRequest form)
    {
        if (form.TenantId == Guid.Empty)
            return BadRequest("TenantId is required.");

        if (form.Image == null || form.Image.Length == 0)
            return BadRequest("Image is required.");

        var allowedExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif" };
        var extension = Path.GetExtension(form.Image.FileName).ToLowerInvariant();

        if (!allowedExtensions.Contains(extension))
            return BadRequest("Invalid image format. Allowed formats: jpg, jpeg, png, bmp, tiff, tif.");

        const long maxFileSize = 5 * 1024 * 1024;

        if (form.Image.Length > maxFileSize)
            return BadRequest("Image size exceeds the maximum allowed size of 5 MB.");

        try
        {
            using var memoryStream = new MemoryStream();
            await form.Image.CopyToAsync(memoryStream);

            var request = new IdentifyRequest
            {
                TenantId = form.TenantId,
                ImageBytes = memoryStream.ToArray()
            };

            var response = await _identificationService.IdentifyAsync(request);

            return Ok(response);
        }
        catch (ArgumentException ex)
        {
            return BadRequest(ex.Message);
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ex.Message);
        }
        catch
        {
            return StatusCode(500, "An unexpected error occurred during identification.");
        }
    }
}